package com.maduro.faceprivacy;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import android.graphics.Bitmap;
import android.graphics.PointF;
import android.media.FaceDetector;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.media.Image;
import android.graphics.Rect;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Plugin-side video processor. The custom part is the YOLO preprocessing/output
 * decoder, NMS, conservative face tracking, and privacy masks. ONNX Runtime is
 * loaded from the plugin assets so the host app does not need to be rebuilt.
 */
public final class FacePrivacyRuntime {
    private static final int MODEL_SIZE = 384;
    private static final int MAX_FACES = 8;
    private static final long CODEC_TIMEOUT_US = 10_000;
    private static final String VIDEO_MIME = "video/avc";

    private static OrtEnvironment environment;
    private static OrtSession cachedSession;
    private static String cachedModelPath;
    private static String cachedProvider = "CPU";
    private static boolean cachedNnapi;
    private static int cachedThreads;
    private static boolean nativeLibrariesLoaded;

    private FacePrivacyRuntime() { }

    /**
     * Load ONNX Runtime from private plugin storage before OrtEnvironment's
     * static initializer calls System.loadLibrary(). Loading through this
     * class keeps both libraries associated with the same DexClassLoader as
     * the ONNX Java classes, and preloads the dependency before the JNI shim.
     */
    public static synchronized String loadNativeLibraries(String runtimePath, String jniPath) {
        if (nativeLibrariesLoaded) return "ONNX native libraries already loaded";
        File runtimeLibrary = new File(runtimePath);
        File jniLibrary = new File(jniPath);
        if (!runtimeLibrary.isFile() || !jniLibrary.isFile()) {
            throw new UnsatisfiedLinkError("ONNX native library file is missing: "
                    + runtimeLibrary.getAbsolutePath() + " / " + jniLibrary.getAbsolutePath());
        }
        System.setProperty("onnxruntime.native.path", runtimeLibrary.getParent());
        try {
            System.load(runtimeLibrary.getAbsolutePath());
        } catch (UnsatisfiedLinkError e) {
            String msg = e.getMessage();
            if (msg == null || !msg.contains("already opened")) {
                throw e;
            }
        }
        try {
            System.load(jniLibrary.getAbsolutePath());
        } catch (UnsatisfiedLinkError e) {
            String msg = e.getMessage();
            if (msg == null || !msg.contains("already opened")) {
                throw e;
            }
        }
        nativeLibrariesLoaded = true;
        return "ONNX native libraries loaded from plugin storage";
    }

    /**
     * Entry point called by the Python plugin through DexClassLoader reflection.
     * Input and output are JSON so the bridge has one stable Java method signature.
     */
    public static String process(String json) throws Exception {
        JSONObject config = new JSONObject(json);
        String input = config.getString("input");
        String output = config.getString("output");
        String model = config.getString("model");
        String mode = config.optString("mode", "pixelate").toLowerCase();
        int detectEvery = clamp(config.optInt("detect_every", 2), 1, 8);
        float confidence = clamp(config.optDouble("confidence", 0.35), 0.05, 0.95);
        float nmsIou = clamp(config.optDouble("nms_iou", 0.45), 0.05, 0.95);
        float padding = clamp(config.optDouble("padding_percent", 25), 0, 100) / 100.0f;
        int holdFrames = clamp(config.optInt("hold_frames", 15), 0, 120);
        int pixelBlock = clamp(config.optInt("pixel_block", 5), 2, 64);
        int blurRadius = clamp(config.optInt("blur_radius", 16), 2, 80);
        int bitrate = clamp(config.optInt("bitrate", 2_500_000), 400_000, 12_000_000);
        int threads = clamp(config.optInt("cpu_threads", 2), 1, 4);
        boolean preferNnapi = config.optBoolean("nnapi", true);
        int maskColor = parseColor(config.optString("mask_color", "#000000"));
        boolean fallbackDetector = config.optBoolean("android_fallback", true);

        if (!("pixelate".equals(mode) || "blur".equals(mode) || "solid".equals(mode))) {
            throw new IllegalArgumentException("mode must be pixelate, blur, or solid");
        }
        File inFile = new File(input);
        if (!inFile.isFile() || inFile.length() <= 0) {
            throw new IOException("The recorded video file is missing or empty");
        }
        File outFile = new File(output);
        File parent = outFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Could not create the output cache directory");
        }
        if (outFile.exists() && !outFile.delete()) {
            throw new IOException("Could not replace an old processed video");
        }

        OrtSession session = getSession(model, preferNnapi, threads);
        String provider = cachedProvider;
        int frameCount;
        try {
            frameCount = processMp4(
                    input, output, session, mode, detectEvery, confidence, nmsIou,
                    padding, holdFrames, pixelBlock, blurRadius, maskColor,
                    bitrate, fallbackDetector
            );
        } catch (Throwable error) {
            if (outFile.exists()) {
                // A partial file is never eligible for sending.
                //noinspection ResultOfMethodCallIgnored
                outFile.delete();
            }
            if (error instanceof Exception) {
                throw (Exception) error;
            }
            throw new RuntimeException(error);
        }

        if (frameCount <= 0 || !outFile.isFile() || outFile.length() <= 0) {
            if (outFile.exists()) {
                //noinspection ResultOfMethodCallIgnored
                outFile.delete();
            }
            throw new IOException("No processed video frames were produced");
        }
        JSONObject result = new JSONObject();
        result.put("frames", frameCount);
        result.put("bytes", outFile.length());
        result.put("provider", provider);
        result.put("output", output);
        return result.toString();
    }

    private static synchronized OrtSession getSession(String model, boolean preferNnapi, int threads) throws Exception {
        if (cachedSession != null && model.equals(cachedModelPath)
                && preferNnapi == cachedNnapi && threads == cachedThreads) {
            return cachedSession;
        }
        if (cachedSession != null) {
            try { cachedSession.close(); } catch (Exception ignored) { }
            cachedSession = null;
        }
        if (environment == null) {
            environment = OrtEnvironment.getEnvironment();
        }

        OrtSession created = null;
        if (preferNnapi) {
            try {
                created = createSession(model, threads, true);
                cachedProvider = "NNAPI (CPU fallback enabled)";
            } catch (Throwable nnapiError) {
                // A device without a usable NNAPI partition still gets the CPU path.
                created = createSession(model, threads, false);
                cachedProvider = "CPU fallback";
            }
        } else {
            created = createSession(model, threads, false);
            cachedProvider = "CPU";
        }
        cachedSession = created;
        cachedModelPath = model;
        cachedNnapi = preferNnapi;
        cachedThreads = threads;
        return cachedSession;
    }

    private static OrtSession createSession(String model, int threads, boolean nnapi) throws Exception {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
            options.setIntraOpNumThreads(threads);
            options.setInterOpNumThreads(1);
            options.setMemoryPatternOptimization(true);
            if (nnapi) {
                options.addNnapi();
            }
            return environment.createSession(model, options);
        }
    }

    private static int processMp4(
            String inputPath,
            String outputPath,
            OrtSession session,
            String mode,
            int detectEvery,
            float confidence,
            float nmsIou,
            float padding,
            int holdFrames,
            int pixelBlock,
            int blurRadius,
            int maskColor,
            int bitrate,
            boolean fallbackDetector
    ) throws Exception {
        MediaExtractor videoExtractor = new MediaExtractor();
        MediaExtractor audioExtractor = null;
        MediaCodec decoder = null;
        MediaCodec encoder = null;
        MediaMuxer muxer = null;
        MuxState muxState = new MuxState();
        boolean decoderStarted = false;
        boolean encoderStarted = false;
        int frames = 0;
        long lastPresentationTimeUs = -1;
        boolean decoderInputDone = false;
        boolean decoderOutputDone = false;
        boolean encoderOutputDone = false;
        int idleLoops = 0;
        Tracker tracker = new Tracker();
        int[] reusablePixels = null;
        FloatBuffer inferenceInput = ByteBuffer.allocateDirect(3 * MODEL_SIZE * MODEL_SIZE * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();

        try {
            videoExtractor.setDataSource(inputPath);
            int videoTrack = findTrack(videoExtractor, "video/");
            if (videoTrack < 0) {
                throw new IOException("The recorded file has no video track");
            }
            videoExtractor.selectTrack(videoTrack);
            MediaFormat inputFormat = videoExtractor.getTrackFormat(videoTrack);
            String inputMime = inputFormat.getString(MediaFormat.KEY_MIME);
            int width = inputFormat.getInteger(MediaFormat.KEY_WIDTH);
            int height = inputFormat.getInteger(MediaFormat.KEY_HEIGHT);
            int frameRate = inputFormat.containsKey(MediaFormat.KEY_FRAME_RATE)
                    ? inputFormat.getInteger(MediaFormat.KEY_FRAME_RATE) : 25;
            frameRate = clamp(frameRate, 1, 60);
            if (width < 16 || height < 16 || width > 1080 || height > 1080) {
                throw new IOException("Unsupported round-video dimensions: " + width + "x" + height);
            }

            audioExtractor = new MediaExtractor();
            audioExtractor.setDataSource(inputPath);
            int audioTrack = findTrack(audioExtractor, "audio/");
            if (audioTrack >= 0) {
                audioExtractor.selectTrack(audioTrack);
            }

            muxer = new MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
            if (audioTrack >= 0) {
                muxState.audioTrack = muxer.addTrack(audioExtractor.getTrackFormat(audioTrack));
                muxState.audioExtractor = audioExtractor;
            }
            muxState.muxer = muxer;

            decoder = MediaCodec.createDecoderByType(inputMime);
            decoder.configure(inputFormat, null, null, 0);
            decoder.start();
            decoderStarted = true;

            MediaFormat outputFormat = MediaFormat.createVideoFormat(VIDEO_MIME, width, height);
            outputFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            outputFormat.setInteger(MediaFormat.KEY_BIT_RATE, bitrate);
            outputFormat.setInteger(MediaFormat.KEY_FRAME_RATE, frameRate);
            outputFormat.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);
            encoder = createYuvEncoder(outputFormat);
            encoder.start();
            encoderStarted = true;

            MediaCodec.BufferInfo decoderInfo = new MediaCodec.BufferInfo();
            MediaCodec.BufferInfo encoderInfo = new MediaCodec.BufferInfo();

            while (!decoderOutputDone) {
                boolean madeProgress = false;

                if (!decoderInputDone) {
                    int inputIndex = decoder.dequeueInputBuffer(CODEC_TIMEOUT_US);
                    if (inputIndex >= 0) {
                        ByteBuffer inputBuffer = decoder.getInputBuffer(inputIndex);
                        if (inputBuffer == null) {
                            throw new IOException("Video decoder returned no input buffer");
                        }
                        inputBuffer.clear();
                        int sampleSize = videoExtractor.readSampleData(inputBuffer, 0);
                        if (sampleSize < 0) {
                            decoder.queueInputBuffer(inputIndex, 0, 0,
                                    Math.max(0, videoExtractor.getSampleTime()),
                                    MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            decoderInputDone = true;
                        } else {
                            long sampleTimeUs = Math.max(0, videoExtractor.getSampleTime());
                            decoder.queueInputBuffer(inputIndex, 0, sampleSize, sampleTimeUs,
                                    videoExtractor.getSampleFlags());
                            videoExtractor.advance();
                        }
                        madeProgress = true;
                    }
                }

                int outputIndex = decoder.dequeueOutputBuffer(decoderInfo, CODEC_TIMEOUT_US);
                if (outputIndex >= 0) {
                    if (decoderInfo.size > 0) {
                        Image image = decoder.getOutputImage(outputIndex);
                        if (image == null) {
                            decoder.releaseOutputBuffer(outputIndex, false);
                            throw new IOException("Video decoder did not expose a YUV frame");
                        }
                        int[] pixels;
                        try {
                            if (reusablePixels == null || reusablePixels.length != width * height) {
                                reusablePixels = new int[width * height];
                            }
                            imageToRgb(image, width, height, reusablePixels);
                            pixels = reusablePixels;
                        } finally {
                            image.close();
                        }

                        if (frames % detectEvery == 0) {
                            List<Box> detections = runYolo(session, inferenceInput, pixels, width, height,
                                    confidence, nmsIou);
                            if (detections.isEmpty() && fallbackDetector) {
                                detections = runAndroidFallback(pixels, width, height);
                            }
                            tracker.update(detections, frames, holdFrames);
                        } else {
                            tracker.advance(frames, holdFrames);
                        }

                        applyMasks(pixels, width, height, tracker, mode, padding,
                                pixelBlock, blurRadius, maskColor, holdFrames);

                        int encoderInputIndex = waitForEncoderInput(encoder, encoderInfo,
                                muxState, bitrate, audioTrack >= 0);
                        Image encoderImage = encoder.getInputImage(encoderInputIndex);
                        if (encoderImage == null) {
                            encoder.queueInputBuffer(encoderInputIndex, 0, 0,
                                    Math.max(0, decoderInfo.presentationTimeUs), 0);
                            decoder.releaseOutputBuffer(outputIndex, false);
                            throw new IOException("Video encoder did not expose a YUV input frame");
                        }
                        long pts = Math.max(0, decoderInfo.presentationTimeUs);
                        if (pts <= lastPresentationTimeUs) {
                            pts = lastPresentationTimeUs + 1;
                        }
                        lastPresentationTimeUs = pts;
                        try {
                            rgbToImage(pixels, width, height, encoderImage);
                            encoder.queueInputBuffer(encoderInputIndex, 0, 0, pts, 0);
                        } finally {
                            encoderImage.close();
                        }
                        frames++;
                    }

                    boolean eos = (decoderInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                    decoder.releaseOutputBuffer(outputIndex, false);
                    decoderOutputDone = eos;
                    madeProgress = true;
                } else if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED
                        || outputIndex == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED) {
                    madeProgress = true;
                }

                encoderOutputDone = drainEncoder(encoder, encoderInfo, muxState,
                        false, bitrate, audioTrack >= 0);
                if (decoderOutputDone) {
                    break;
                }
                if (madeProgress) {
                    idleLoops = 0;
                } else if (++idleLoops > 500) {
                    throw new IOException("Video codec made no progress");
                }
            }

            int eosIndex;
            do {
                eosIndex = encoder.dequeueInputBuffer(CODEC_TIMEOUT_US);
                if (eosIndex < 0) {
                    drainEncoder(encoder, encoderInfo, muxState, false, bitrate, audioTrack >= 0);
                }
            } while (eosIndex < 0);
            long eosPts = Math.max(0, lastPresentationTimeUs + 1_000_000L / frameRate);
            encoder.queueInputBuffer(eosIndex, 0, 0, eosPts,
                    MediaCodec.BUFFER_FLAG_END_OF_STREAM);

            int eosDrainLoops = 0;
            while (!encoderOutputDone && eosDrainLoops++ < 500) {
                encoderOutputDone = drainEncoder(encoder, encoderInfo, muxState,
                        true, bitrate, audioTrack >= 0);
            }
            if (!muxState.started) {
                throw new IOException("Video encoder produced no muxable output format");
            }
            if (!encoderOutputDone) {
                throw new IOException("Video encoder did not finish");
            }
            return frames;
        } finally {
            if (decoder != null) {
                try { if (decoderStarted) decoder.stop(); } catch (Exception ignored) { }
                try { decoder.release(); } catch (Exception ignored) { }
            }
            if (encoder != null) {
                try { if (encoderStarted) encoder.stop(); } catch (Exception ignored) { }
                try { encoder.release(); } catch (Exception ignored) { }
            }
            if (muxer != null) {
                try { if (muxState.started) muxer.stop(); } catch (Exception ignored) { }
                try { muxer.release(); } catch (Exception ignored) { }
            }
            try { videoExtractor.release(); } catch (Exception ignored) { }
            if (audioExtractor != null) {
                try { audioExtractor.release(); } catch (Exception ignored) { }
            }
        }
    }

    private static MediaCodec createYuvEncoder(MediaFormat format) throws Exception {
        MediaCodecList codecList = new MediaCodecList(MediaCodecList.REGULAR_CODECS);
        for (MediaCodecInfo info : codecList.getCodecInfos()) {
            if (!info.isEncoder()) continue;
            boolean supportsAvc = false;
            for (String type : info.getSupportedTypes()) {
                if (VIDEO_MIME.equalsIgnoreCase(type)) {
                    supportsAvc = true;
                    break;
                }
            }
            if (!supportsAvc) continue;
            MediaCodecInfo.CodecCapabilities capabilities = info.getCapabilitiesForType(VIDEO_MIME);
            boolean flexible = false;
            for (int color : capabilities.colorFormats) {
                if (color == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible) {
                    flexible = true;
                    break;
                }
            }
            if (!flexible) continue;
            MediaCodec candidate = null;
            try {
                candidate = MediaCodec.createByCodecName(info.getName());
                candidate.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
                return candidate;
            } catch (Exception ignored) {
                if (candidate != null) {
                    try { candidate.release(); } catch (Exception ignoredAgain) { }
                }
            }
        }
        throw new IOException("No H.264 encoder accepts flexible YUV input on this device");
    }

    private static int waitForEncoderInput(MediaCodec encoder,
                                           MediaCodec.BufferInfo info,
                                           MuxState state,
                                           int bitrate,
                                           boolean hasAudio) throws Exception {
        int index;
        int loops = 0;
        while ((index = encoder.dequeueInputBuffer(CODEC_TIMEOUT_US)) < 0) {
            drainEncoder(encoder, info, state, false, bitrate, hasAudio);
            if (++loops > 500) {
                throw new IOException("Video encoder input queue stalled");
            }
        }
        return index;
    }

    private static boolean drainEncoder(MediaCodec encoder,
                                        MediaCodec.BufferInfo info,
                                        MuxState state,
                                        boolean waitForOutput,
                                        int bitrate,
                                        boolean hasAudio) throws Exception {
        long timeout = waitForOutput ? CODEC_TIMEOUT_US : 0;
        while (true) {
            int status = encoder.dequeueOutputBuffer(info, timeout);
            timeout = 0;
            if (status == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (state.started) {
                    throw new IOException("Encoder output format changed after muxer start");
                }
                state.videoTrack = state.muxer.addTrack(encoder.getOutputFormat());
                state.muxer.start();
                state.started = true;
                if (hasAudio && !state.audioCopied) {
                    copyAudioTrack(state.audioExtractor, state.muxer, state.audioTrack);
                    state.audioCopied = true;
                }
                continue;
            }
            if (status == MediaCodec.INFO_TRY_AGAIN_LATER) {
                return false;
            }
            if (status == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED) {
                continue;
            }
            if (status < 0) {
                continue;
            }

            boolean eos = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
            ByteBuffer encoded = encoder.getOutputBuffer(status);
            if ((info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                info.size = 0;
            }
            if (info.size > 0) {
                if (!state.started || encoded == null) {
                    encoder.releaseOutputBuffer(status, false);
                    throw new IOException("Encoded frame arrived before muxer start");
                }
                encoded.position(info.offset);
                encoded.limit(info.offset + info.size);
                state.muxer.writeSampleData(state.videoTrack, encoded, info);
            }
            encoder.releaseOutputBuffer(status, false);
            if (eos) {
                return true;
            }
        }
    }

    private static void copyAudioTrack(MediaExtractor extractor,
                                       MediaMuxer muxer,
                                       int muxerTrack) throws IOException {
        if (extractor == null || muxerTrack < 0) return;
        long sampleSize = extractor.getSampleSize();
        int capacity = (int) Math.min(Math.max(sampleSize, 256 * 1024L), 8 * 1024 * 1024L);
        ByteBuffer buffer = ByteBuffer.allocateDirect(capacity);
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        while (sampleSize >= 0) {
            if (sampleSize > buffer.capacity()) {
                buffer = ByteBuffer.allocateDirect((int) Math.min(sampleSize, 8 * 1024 * 1024L));
            }
            buffer.clear();
            int size = extractor.readSampleData(buffer, 0);
            if (size < 0) break;
            info.set(0, size, Math.max(0, extractor.getSampleTime()), extractor.getSampleFlags());
            muxer.writeSampleData(muxerTrack, buffer, info);
            if (!extractor.advance()) break;
            sampleSize = extractor.getSampleSize();
        }
    }

    private static int findTrack(MediaExtractor extractor, String prefix) {
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            MediaFormat format = extractor.getTrackFormat(i);
            String mime = format.getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith(prefix)) return i;
        }
        return -1;
    }

    private static void imageToRgb(Image image, int width, int height, int[] pixels) throws IOException {
        Image.Plane[] planes = image.getPlanes();
        if (planes == null || planes.length < 3) {
            throw new IOException("Decoder frame is not YUV_420_888");
        }
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();
        Rect crop = image.getCropRect();
        int cropLeft = crop == null ? 0 : crop.left;
        int cropTop = crop == null ? 0 : crop.top;
        int cropWidth = crop == null ? width : crop.width();
        int cropHeight = crop == null ? height : crop.height();
        int yRow = planes[0].getRowStride();
        int yPixel = planes[0].getPixelStride();
        int uRow = planes[1].getRowStride();
        int uPixel = planes[1].getPixelStride();
        int vRow = planes[2].getRowStride();
        int vPixel = planes[2].getPixelStride();

        for (int y = 0; y < height; y++) {
            int sy = cropTop + Math.min(cropHeight - 1, y * cropHeight / height);
            int uvY = sy >> 1;
            for (int x = 0; x < width; x++) {
                int sx = cropLeft + Math.min(cropWidth - 1, x * cropWidth / width);
                int yy = yBuffer.get(sy * yRow + sx * yPixel) & 0xff;
                int u = (uBuffer.get(uvY * uRow + (sx >> 1) * uPixel) & 0xff) - 128;
                int v = (vBuffer.get(uvY * vRow + (sx >> 1) * vPixel) & 0xff) - 128;
                int c = Math.max(0, yy - 16);
                int r = clamp((298 * c + 409 * v + 128) >> 8, 0, 255);
                int g = clamp((298 * c - 100 * u - 208 * v + 128) >> 8, 0, 255);
                int b = clamp((298 * c + 516 * u + 128) >> 8, 0, 255);
                pixels[y * width + x] = 0xff000000 | (r << 16) | (g << 8) | b;
            }
        }
    }

    private static void rgbToImage(int[] pixels, int width, int height, Image image) throws IOException {
        Image.Plane[] planes = image.getPlanes();
        if (planes == null || planes.length < 3) {
            throw new IOException("Encoder frame is not YUV_420_888");
        }
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();
        int yRow = planes[0].getRowStride();
        int yPixel = planes[0].getPixelStride();
        int uRow = planes[1].getRowStride();
        int uPixel = planes[1].getPixelStride();
        int vRow = planes[2].getRowStride();
        int vPixel = planes[2].getPixelStride();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int color = pixels[y * width + x];
                int r = (color >> 16) & 0xff;
                int g = (color >> 8) & 0xff;
                int b = color & 0xff;
                int yy = clamp(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16, 0, 255);
                yBuffer.put(y * yRow + x * yPixel, (byte) yy);
            }
        }
        for (int y = 0; y < height; y += 2) {
            for (int x = 0; x < width; x += 2) {
                int rSum = 0, gSum = 0, bSum = 0, n = 0;
                for (int dy = 0; dy < 2 && y + dy < height; dy++) {
                    for (int dx = 0; dx < 2 && x + dx < width; dx++) {
                        int color = pixels[(y + dy) * width + x + dx];
                        rSum += (color >> 16) & 0xff;
                        gSum += (color >> 8) & 0xff;
                        bSum += color & 0xff;
                        n++;
                    }
                }
                int r = rSum / n, g = gSum / n, b = bSum / n;
                int u = clamp(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128, 0, 255);
                int v = clamp(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128, 0, 255);
                int uvX = x >> 1;
                int uvY = y >> 1;
                uBuffer.put(uvY * uRow + uvX * uPixel, (byte) u);
                vBuffer.put(uvY * vRow + uvX * vPixel, (byte) v);
            }
        }
    }

    private static List<Box> runYolo(OrtSession session,
                                     FloatBuffer direct,
                                     int[] pixels,
                                     int width,
                                     int height,
                                     float confidence,
                                     float nmsIou) throws Exception {
        int plane = MODEL_SIZE * MODEL_SIZE;
        direct.clear();
        for (int y = 0; y < MODEL_SIZE; y++) {
            int sy = Math.min(height - 1, y * height / MODEL_SIZE);
            for (int x = 0; x < MODEL_SIZE; x++) {
                int sx = Math.min(width - 1, x * width / MODEL_SIZE);
                int color = pixels[sy * width + sx];
                int index = y * MODEL_SIZE + x;
                direct.put(index, ((color >> 16) & 0xff) / 255.0f);
                direct.put(plane + index, ((color >> 8) & 0xff) / 255.0f);
                direct.put(plane * 2 + index, (color & 0xff) / 255.0f);
            }
        }
        direct.position(0);
        direct.limit(plane * 3);
        String inputName = session.getInputNames().iterator().next();
        long[] shape = new long[]{1, 3, MODEL_SIZE, MODEL_SIZE};
        try (OnnxTensor tensor = OnnxTensor.createTensor(
                environment, direct, shape)) {
            Map<String, OnnxTensor> inputMap = new HashMap<>();
            inputMap.put(inputName, tensor);
            try (OrtSession.Result result = session.run(inputMap)) {
                if (result.size() < 1) throw new IOException("YOLO returned no output tensor");
                OnnxValue value = result.get(0);
                if (!(value.getInfo() instanceof TensorInfo)) {
                    throw new IOException("YOLO output is not a tensor");
                }
                long[] outputShape = ((TensorInfo) value.getInfo()).getShape();
                Object raw = value.getValue();
                if (!(raw instanceof float[][][])) {
                    throw new IOException("YOLO output tensor has an unsupported type");
                }
                return decodeYolo((float[][][]) raw, outputShape, width, height,
                        confidence, nmsIou);
            }
        }
    }

    private static List<Box> decodeYolo(float[][][] values,
                                        long[] shape,
                                        int width,
                                        int height,
                                        float threshold,
                                        float iouThreshold) throws IOException {
        List<Box> candidates = new ArrayList<>();
        if (values.length == 0 || values[0].length == 0) return candidates;
        int dim1 = values[0].length;
        int dim2 = values[0][0].length;
        boolean channelsFirst = dim1 <= 128 && dim2 > dim1;
        int features = channelsFirst ? dim1 : dim2;
        int count = channelsFirst ? dim2 : dim1;
        if (features < 5) {
            throw new IOException("YOLO output has fewer than 5 values per box");
        }

        for (int i = 0; i < count; i++) {
            float cx = channelsFirst ? values[0][0][i] : values[0][i][0];
            float cy = channelsFirst ? values[0][1][i] : values[0][i][1];
            float bw = channelsFirst ? values[0][2][i] : values[0][i][2];
            float bh = channelsFirst ? values[0][3][i] : values[0][i][3];
            float score = 0;
            if (features == 5) {
                score = channelsFirst ? values[0][4][i] : values[0][i][4];
            } else {
                for (int c = 4; c < features; c++) {
                    score = Math.max(score, channelsFirst ? values[0][c][i] : values[0][i][c]);
                }
            }
            if (!finite(cx) || !finite(cy) || !finite(bw) || !finite(bh)
                    || !finite(score) || score < threshold || bw <= 1 || bh <= 1) {
                continue;
            }
            if (Math.max(Math.max(Math.abs(cx), Math.abs(cy)), Math.max(bw, bh)) <= 2.0f) {
                cx *= MODEL_SIZE;
                cy *= MODEL_SIZE;
                bw *= MODEL_SIZE;
                bh *= MODEL_SIZE;
            }
            float scaleX = width / (float) MODEL_SIZE;
            float scaleY = height / (float) MODEL_SIZE;
            float x1 = clamp((cx - bw * 0.5f) * scaleX, 0, width - 1);
            float y1 = clamp((cy - bh * 0.5f) * scaleY, 0, height - 1);
            float x2 = clamp((cx + bw * 0.5f) * scaleX, 0, width - 1);
            float y2 = clamp((cy + bh * 0.5f) * scaleY, 0, height - 1);
            if (x2 > x1 && y2 > y1) candidates.add(new Box(x1, y1, x2, y2, score));
        }

        Collections.sort(candidates, new Comparator<Box>() {
            @Override public int compare(Box a, Box b) {
                return Float.compare(b.score, a.score);
            }
        });
        List<Box> selected = new ArrayList<>();
        for (Box candidate : candidates) {
            boolean suppressed = false;
            for (Box previous : selected) {
                if (iou(candidate, previous) > iouThreshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                selected.add(candidate);
                if (selected.size() >= MAX_FACES) break;
            }
        }
        return selected;
    }

    private static List<Box> runAndroidFallback(int[] pixels, int width, int height) {
        List<Box> boxes = new ArrayList<>();
        if ((width & 1) != 0 || width < 32 || height < 32) return boxes;
        Bitmap bitmap = null;
        try {
            bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
            bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
            FaceDetector.Face[] faces = new FaceDetector.Face[4];
            FaceDetector detector = new FaceDetector(width, height, faces.length);
            int found = detector.findFaces(bitmap, faces);
            for (int i = 0; i < found; i++) {
                FaceDetector.Face face = faces[i];
                if (face == null || face.confidence() < 0.25f) continue;
                PointF midpoint = new PointF();
                face.getMidPoint(midpoint);
                float eyes = face.eyesDistance();
                if (eyes < 2) continue;
                float bw = eyes * 2.8f;
                float bh = eyes * 3.2f;
                float cx = midpoint.x;
                float cy = midpoint.y + eyes * 0.55f;
                boxes.add(new Box(
                        clamp(cx - bw / 2, 0, width - 1),
                        clamp(cy - bh / 2, 0, height - 1),
                        clamp(cx + bw / 2, 0, width - 1),
                        clamp(cy + bh / 2, 0, height - 1),
                        face.confidence()
                ));
            }
        } catch (Throwable ignored) {
            // The YOLO tracks remain active if the platform fallback is unavailable.
        } finally {
            if (bitmap != null) bitmap.recycle();
        }
        return boxes;
    }

    private static void applyMasks(int[] pixels,
                                   int width,
                                   int height,
                                   Tracker tracker,
                                   String mode,
                                   float padding,
                                   int pixelBlock,
                                   int blurRadius,
                                   int maskColor,
                                   int holdFrames) {
        for (Track track : tracker.tracks) {
            int age = Math.max(0, tracker.currentFrame - track.lastSeenFrame);
            if (age > holdFrames) continue;
            float extra = Math.min(0.50f, age * 0.0125f);
            float padX = track.w * (padding + extra);
            float padY = track.h * (padding + extra);
            int x1 = clamp((int) Math.floor(track.x - padX), 0, width - 1);
            int y1 = clamp((int) Math.floor(track.y - padY), 0, height - 1);
            int x2 = clamp((int) Math.ceil(track.x + track.w + padX), 0, width);
            int y2 = clamp((int) Math.ceil(track.y + track.h + padY), 0, height);
            if (x2 <= x1 || y2 <= y1) continue;
            if ("solid".equals(mode)) {
                fillRect(pixels, width, x1, y1, x2, y2, maskColor);
            } else if ("blur".equals(mode)) {
                blurRect(pixels, width, x1, y1, x2, y2, blurRadius);
            } else {
                pixelateRect(pixels, width, x1, y1, x2, y2, pixelBlock);
            }
        }
    }

    private static void fillRect(int[] pixels, int width, int x1, int y1, int x2, int y2, int color) {
        for (int y = y1; y < y2; y++) {
            int start = y * width + x1;
            for (int x = x1; x < x2; x++) pixels[start + x - x1] = color;
        }
    }

    private static void pixelateRect(int[] pixels,
                                     int width,
                                     int x1,
                                     int y1,
                                     int x2,
                                     int y2,
                                     int block) {
        for (int by = y1; by < y2; by += block) {
            int ey = Math.min(y2, by + block);
            for (int bx = x1; bx < x2; bx += block) {
                int ex = Math.min(x2, bx + block);
                long r = 0, g = 0, b = 0, n = 0;
                for (int y = by; y < ey; y++) {
                    int base = y * width;
                    for (int x = bx; x < ex; x++) {
                        int color = pixels[base + x];
                        r += (color >> 16) & 0xff;
                        g += (color >> 8) & 0xff;
                        b += color & 0xff;
                        n++;
                    }
                }
                int average = 0xff000000 | ((int) (r / n) << 16)
                        | ((int) (g / n) << 8) | (int) (b / n);
                for (int y = by; y < ey; y++) {
                    int base = y * width;
                    for (int x = bx; x < ex; x++) pixels[base + x] = average;
                }
            }
        }
    }

    private static void blurRect(int[] pixels,
                                 int width,
                                 int x1,
                                 int y1,
                                 int x2,
                                 int y2,
                                 int radius) {
        int rw = x2 - x1;
        int rh = y2 - y1;
        if (rw < 3 || rh < 3) return;
        int[] region = new int[rw * rh];
        for (int y = 0; y < rh; y++) {
            System.arraycopy(pixels, (y1 + y) * width + x1, region, y * rw, rw);
        }
        int r = Math.min(radius, Math.max(1, Math.min(rw, rh) / 2));
        for (int i = 0; i < 3; i++) region = boxBlur(region, rw, rh, r);
        for (int y = 0; y < rh; y++) {
            System.arraycopy(region, y * rw, pixels, (y1 + y) * width + x1, rw);
        }
    }

    private static int[] boxBlur(int[] input, int width, int height, int radius) {
        int[] horizontal = new int[input.length];
        int[] output = new int[input.length];
        for (int y = 0; y < height; y++) {
            long sr = 0, sg = 0, sb = 0;
            int count = 0;
            for (int x = -radius; x <= radius; x++) {
                int sx = clamp(x, 0, width - 1);
                int c = input[y * width + sx];
                sr += (c >> 16) & 0xff; sg += (c >> 8) & 0xff; sb += c & 0xff; count++;
            }
            for (int x = 0; x < width; x++) {
                horizontal[y * width + x] = 0xff000000 | ((int) (sr / count) << 16)
                        | ((int) (sg / count) << 8) | (int) (sb / count);
                int removeX = clamp(x - radius, 0, width - 1);
                int addX = clamp(x + radius + 1, 0, width - 1);
                int remove = input[y * width + removeX];
                int add = input[y * width + addX];
                sr += ((add >> 16) & 0xff) - ((remove >> 16) & 0xff);
                sg += ((add >> 8) & 0xff) - ((remove >> 8) & 0xff);
                sb += (add & 0xff) - (remove & 0xff);
            }
        }
        for (int x = 0; x < width; x++) {
            long sr = 0, sg = 0, sb = 0;
            int count = 0;
            for (int y = -radius; y <= radius; y++) {
                int sy = clamp(y, 0, height - 1);
                int c = horizontal[sy * width + x];
                sr += (c >> 16) & 0xff; sg += (c >> 8) & 0xff; sb += c & 0xff; count++;
            }
            for (int y = 0; y < height; y++) {
                output[y * width + x] = 0xff000000 | ((int) (sr / count) << 16)
                        | ((int) (sg / count) << 8) | (int) (sb / count);
                int removeY = clamp(y - radius, 0, height - 1);
                int addY = clamp(y + radius + 1, 0, height - 1);
                int remove = horizontal[removeY * width + x];
                int add = horizontal[addY * width + x];
                sr += ((add >> 16) & 0xff) - ((remove >> 16) & 0xff);
                sg += ((add >> 8) & 0xff) - ((remove >> 8) & 0xff);
                sb += (add & 0xff) - (remove & 0xff);
            }
        }
        return output;
    }

    private static int parseColor(String value) {
        try {
            String hex = value == null ? "#000000" : value.trim();
            if (!hex.startsWith("#")) hex = "#" + hex;
            long color = Long.parseLong(hex.substring(1), 16);
            if (hex.length() == 7) color |= 0xff000000L;
            return (int) color;
        } catch (Exception ignored) {
            return 0xff000000;
        }
    }

    private static float iou(Box a, Box b) {
        float x1 = Math.max(a.x1, b.x1);
        float y1 = Math.max(a.y1, b.y1);
        float x2 = Math.min(a.x2, b.x2);
        float y2 = Math.min(a.y2, b.y2);
        float intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        float union = a.area() + b.area() - intersection;
        return union <= 0 ? 0 : intersection / union;
    }

    private static float iou(Track a, Box b) {
        return iou(new Box(a.x, a.y, a.x + a.w, a.y + a.h, 1), b);
    }

    private static boolean finite(float x) {
        return !Float.isNaN(x) && !Float.isInfinite(x);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static float clamp(double value, double min, double max) {
        return (float) Math.max(min, Math.min(max, value));
    }

    private static final class MuxState {
        MediaMuxer muxer;
        MediaExtractor audioExtractor;
        int videoTrack = -1;
        int audioTrack = -1;
        boolean started;
        boolean audioCopied;
    }

    private static final class Box {
        final float x1, y1, x2, y2, score;
        Box(float x1, float y1, float x2, float y2, float score) {
            this.x1 = x1; this.y1 = y1; this.x2 = x2; this.y2 = y2; this.score = score;
        }
        float area() { return Math.max(0, x2 - x1) * Math.max(0, y2 - y1); }
    }

    private static final class Track {
        float x, y, w, h;
        float vx, vy, vw, vh;
        float lastSeenX, lastSeenY, lastSeenW, lastSeenH;
        int lastSeenFrame;
        int lastUpdateFrame;
        Track(Box box, int frame) {
            x = box.x1; y = box.y1; w = box.x2 - box.x1; h = box.y2 - box.y1;
            lastSeenX = x; lastSeenY = y; lastSeenW = w; lastSeenH = h;
            lastSeenFrame = frame; lastUpdateFrame = frame;
        }
    }

    private static final class Tracker {
        final List<Track> tracks = new ArrayList<>();
        int currentFrame;

        void advance(int frame, int holdFrames) {
            currentFrame = frame;
            for (Track track : tracks) {
                int dt = frame - track.lastUpdateFrame;
                if (dt > 0) {
                    track.x += track.vx * dt;
                    track.y += track.vy * dt;
                    track.w = Math.max(4, track.w + track.vw * dt);
                    track.h = Math.max(4, track.h + track.vh * dt);
                    track.lastUpdateFrame = frame;
                }
            }
            removeExpired(frame, holdFrames);
        }

        void update(List<Box> detections, int frame, int holdFrames) {
            advance(frame, holdFrames);
            boolean[] matched = new boolean[tracks.size()];
            for (Box detection : detections) {
                int bestIndex = -1;
                float bestScore = 0.08f;
                for (int i = 0; i < tracks.size(); i++) {
                    if (matched[i]) continue;
                    Track track = tracks.get(i);
                    float overlap = iou(track, detection);
                    float dcx = (detection.x1 + detection.x2) * 0.5f;
                    float dcy = (detection.y1 + detection.y2) * 0.5f;
                    float tcx = track.x + track.w * 0.5f;
                    float tcy = track.y + track.h * 0.5f;
                    float maxDim = Math.max(track.w, track.h) * 1.25f;
                    float distance = (float) Math.hypot(dcx - tcx, dcy - tcy);
                    float score = overlap;
                    if (distance < maxDim) score = Math.max(score, 0.15f + 0.5f * (1 - distance / maxDim));
                    if (score > bestScore) {
                        bestScore = score;
                        bestIndex = i;
                    }
                }
                if (bestIndex < 0) {
                    tracks.add(new Track(detection, frame));
                    boolean[] expanded = new boolean[tracks.size()];
                    System.arraycopy(matched, 0, expanded, 0, matched.length);
                    expanded[expanded.length - 1] = true;
                    matched = expanded;
                    continue;
                }
                matched[bestIndex] = true;
                Track track = tracks.get(bestIndex);
                float newX = detection.x1;
                float newY = detection.y1;
                float newW = detection.x2 - detection.x1;
                float newH = detection.y2 - detection.y1;
                int dt = Math.max(1, frame - track.lastSeenFrame);
                float measuredVx = (newX - track.lastSeenX) / dt;
                float measuredVy = (newY - track.lastSeenY) / dt;
                float measuredVw = (newW - track.lastSeenW) / dt;
                float measuredVh = (newH - track.lastSeenH) / dt;
                track.vx = 0.65f * track.vx + 0.35f * measuredVx;
                track.vy = 0.65f * track.vy + 0.35f * measuredVy;
                track.vw = 0.65f * track.vw + 0.35f * measuredVw;
                track.vh = 0.65f * track.vh + 0.35f * measuredVh;
                track.x = newX; track.y = newY; track.w = newW; track.h = newH;
                track.lastSeenX = newX; track.lastSeenY = newY;
                track.lastSeenW = newW; track.lastSeenH = newH;
                track.lastSeenFrame = frame; track.lastUpdateFrame = frame;
            }
            removeExpired(frame, holdFrames);
        }

        private void removeExpired(int frame, int holdFrames) {
            for (int i = tracks.size() - 1; i >= 0; i--) {
                if (frame - tracks.get(i).lastSeenFrame > holdFrames) tracks.remove(i);
            }
        }
    }
}
