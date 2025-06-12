package com.example.mobile;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import android.Manifest;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class TFCamera extends AppCompatActivity {
    private static final String TAG = "TFCamera";
    private PreviewView previewView;
    private Interpreter tflite;
    private TextView outputText;
    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;
    private static final int REQUEST_CAMERA_PERMISSION = 10;

    // Model parameters - UPDATE THESE
    private static final int IMG_SIZE = 128;  // Changed from 64 to 128
    private static final int SEQUENCE_LENGTH = 8;  // Number of frames needed

    // Frame buffer to store sequence of frames
    private LinkedList<float[][][]> frameBuffer = new LinkedList<>();
    private final Object frameBufferLock = new Object();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        previewView = findViewById(R.id.previewView);
        outputText = findViewById(R.id.outputText);
        Intent i = getIntent();
        String modelName = i.getStringExtra("model");
        // Load model
        try {

            Log.d(TAG, "Attempting to load model: " + modelName + ".tflite");
            tflite = new Interpreter(loadModelFile(modelName + ".tflite"));
            printModelInfo();
            Toast.makeText(this, "Model loaded successfully", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "Model failed to load: " + e.getMessage());
            Toast.makeText(this, "Model couldn't be loaded", Toast.LENGTH_SHORT).show();
        }

        cameraExecutor = Executors.newSingleThreadExecutor();

        // Check for camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            if(modelName.equals("base")){
                startCamera3d();
            } else {
                startCamera5d();
            }

        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA},
                    REQUEST_CAMERA_PERMISSION);
        }
    }
    private void startCamera3d() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(IMG_SIZE, IMG_SIZE))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, image -> {
                    try {
                        Bitmap bitmap = toBitmap(image);
                        if (bitmap != null) {
                            runModel(bitmap);
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "Error processing image", e);
                    } finally {
                        image.close();  // ALWAYS close the image
                    }
                });

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageAnalysis);

                preview.setSurfaceProvider(previewView.getSurfaceProvider());

            } catch (Exception e) {
                Log.e(TAG, "Error starting camera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }
    private void runModel(Bitmap bitmap) {
        try {
            if (tflite == null) {
                Log.e(TAG, "Model not loaded");
                return;
            }

            // Get the actual input and output shapes
            int[] inputShape = tflite.getInputTensor(0).shape();
            int[] outputShape = tflite.getOutputTensor(0).shape();

            Log.d(TAG, "Input shape: " + java.util.Arrays.toString(inputShape));
            Log.d(TAG, "Output shape: " + java.util.Arrays.toString(outputShape));

            // Create input based on actual shape (simplified approach)
            Object input;
            if (inputShape.length == 4) {
                // Standard CNN input [batch, height, width, channels]
                Bitmap resized = Bitmap.createScaledBitmap(bitmap, inputShape[2], inputShape[1], true);
                float[][][][] inputArray = new float[inputShape[0]][inputShape[1]][inputShape[2]][inputShape[3]];

                for (int y = 0; y < inputShape[1]; y++) {
                    for (int x = 0; x < inputShape[2]; x++) {
                        int pixel = resized.getPixel(x, y);
                        inputArray[0][y][x][0] = ((pixel >> 16) & 0xFF) / 255.0f;
                        if (inputShape[3] > 1) inputArray[0][y][x][1] = ((pixel >> 8) & 0xFF) / 255.0f;
                        if (inputShape[3] > 2) inputArray[0][y][x][2] = (pixel & 0xFF) / 255.0f;
                    }
                }
                input = inputArray;
            } else {
                Log.e(TAG, "Unsupported input shape for this quick fix");
                return;
            }

            // Create output based on actual shape
            Object output;
            if (outputShape.length == 2) {
                output = new float[outputShape[0]][outputShape[1]];
            } else if (outputShape.length == 1) {
                output = new float[outputShape[0]];
            } else {
                Log.e(TAG, "Unsupported output shape");
                return;
            }

            // Run inference
            tflite.run(input, output);

            // Extract result
            float result;
            if (output instanceof float[][]) {
                result = ((float[][])output)[0][0];
            } else {
                result = ((float[])output)[0];
            }

            // Update UI
            runOnUiThread(() -> {
                changeTextViewColor(result);
            });

        } catch (Exception e) {
            Log.e(TAG, "Error in runModel: " + e.getMessage(), e);
        }
    }
    private void changeTextViewColor(float result){
        outputText.setTextColor(Color.parseColor(result > 0.5 ? "#A24E4E" : "#527E3B"));
        outputText.setBackgroundColor(Color.parseColor(result > 0.5 ? "#EE8576" : "#A8DC8A"));
        String resString = String.format("%.4f", result);
        String vd =  result > 0.5 ? "  Violence detected":"  No violence detected";
        String mixed = resString + vd;
        outputText.setText(mixed);
    }
    private void startCamera5d() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(IMG_SIZE, IMG_SIZE))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, image -> {
                    try {
                        if (!isFinishing() && !isDestroyed()) {
                            Bitmap bitmap = toBitmap(image);
                            if (bitmap != null) {
                                processFrame(bitmap);
                            }
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "Error processing image", e);
                    } finally {
                        image.close();
                    }
                });

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageAnalysis);

                preview.setSurfaceProvider(previewView.getSurfaceProvider());

            } catch (Exception e) {
                Log.e(TAG, "Error starting camera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void processFrame(Bitmap bitmap) {
        try {
            // Convert bitmap to normalized float array
            Bitmap resized = Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true);
            float[][][] frameData = new float[IMG_SIZE][IMG_SIZE][3];

            // Extract RGB values and normalize
            for (int y = 0; y < IMG_SIZE; y++) {
                for (int x = 0; x < IMG_SIZE; x++) {
                    int pixel = resized.getPixel(x, y);
                    frameData[y][x][0] = ((pixel >> 16) & 0xFF) / 255.0f;  // R
                    frameData[y][x][1] = ((pixel >> 8) & 0xFF) / 255.0f;   // G
                    frameData[y][x][2] = (pixel & 0xFF) / 255.0f;          // B
                }
            }

            // Add frame to buffer
            synchronized (frameBufferLock) {
                frameBuffer.addLast(frameData);

                // Keep only the last SEQUENCE_LENGTH frames
                while (frameBuffer.size() > SEQUENCE_LENGTH) {
                    frameBuffer.removeFirst();
                }

                // Run model only when we have enough frames
                if (frameBuffer.size() == SEQUENCE_LENGTH) {
                    runVideoModel();
                }
            }

        } catch (Exception e) {
            Log.e(TAG, "Error processing frame: " + e.getMessage(), e);
        }
    }

    private void runVideoModel() {
        try {
            if (tflite == null) {
                Log.e(TAG, "Model not loaded");
                return;
            }

            if (isFinishing() || isDestroyed()) {
                return;
            }

            // Get model shapes
            int[] inputShape = tflite.getInputTensor(0).shape();
            int[] outputShape = tflite.getOutputTensor(0).shape();

            // Create input tensor: [1, 8, 128, 128, 3]
            float[][][][][] input = new float[1][SEQUENCE_LENGTH][IMG_SIZE][IMG_SIZE][3];

            synchronized (frameBufferLock) {
                // Copy frames from buffer to input tensor
                for (int i = 0; i < SEQUENCE_LENGTH; i++) {
                    float[][][] frame = frameBuffer.get(i);
                    for (int y = 0; y < IMG_SIZE; y++) {
                        for (int x = 0; x < IMG_SIZE; x++) {
                            for (int c = 0; c < 3; c++) {
                                input[0][i][y][x][c] = frame[y][x][c];
                            }
                        }
                    }
                }
            }

            // Create output tensor based on actual output shape
            Object output;
            if (outputShape.length == 2) {
                output = new float[outputShape[0]][outputShape[1]];
            } else if (outputShape.length == 1) {
                output = new float[outputShape[0]];
            } else {
                Log.e(TAG, "Unsupported output shape: " + java.util.Arrays.toString(outputShape));
                return;
            }

            // Run inference
            tflite.run(input, output);

            // Extract result
            float result;
            if (output instanceof float[][]) {
                result = ((float[][])output)[0][0];
            } else {
                result = ((float[])output)[0];
            }

            // Update UI
            if (!isFinishing() && !isDestroyed()) {
                runOnUiThread(() -> {
                    if (outputText != null) {
                        changeTextViewColor(result);
                    }
                });
            }

        } catch (Exception e) {
            Log.e(TAG, "Error in runVideoModel: " + e.getMessage(), e);
        }
    }

    // Your existing methods remain the same...
    private MappedByteBuffer loadModelFile(String filename) throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(filename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        MappedByteBuffer mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

        inputStream.close();
        fileDescriptor.close();

        return mappedByteBuffer;
    }

    private void printModelInfo() {
        if (tflite != null) {
            try {
                int inputTensorCount = tflite.getInputTensorCount();
                int outputTensorCount = tflite.getOutputTensorCount();

                Log.d(TAG, "Input tensor count: " + inputTensorCount);
                Log.d(TAG, "Output tensor count: " + outputTensorCount);

                for (int i = 0; i < inputTensorCount; i++) {
                    int[] inputShape = tflite.getInputTensor(i).shape();
                    Log.d(TAG, "Input tensor " + i + " shape: " + java.util.Arrays.toString(inputShape));
                }

                for (int i = 0; i < outputTensorCount; i++) {
                    int[] outputShape = tflite.getOutputTensor(i).shape();
                    Log.d(TAG, "Output tensor " + i + " shape: " + java.util.Arrays.toString(outputShape));
                }
            } catch (Exception e) {
                Log.e(TAG, "Error getting model info: " + e.getMessage());
            }
        }
    }

    private void stopCamera() {
        if (cameraProvider != null) {
            try {
                cameraProvider.unbindAll();
                Log.d(TAG, "Camera stopped successfully");
            } catch (Exception e) {
                Log.e(TAG, "Error stopping camera", e);
            }
        }
    }

    @Override
    protected void onStop() {
        super.onStop();
        stopCamera();
    }

    @Override
    protected void onDestroy() {
        stopCamera();

        if (tflite != null) {
            tflite.close();
            tflite = null;
        }

        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }

        super.onDestroy();
    }

    // Include your existing toBitmap and yuv420ToNV21 methods here...
    private Bitmap toBitmap(ImageProxy image) {
        // Convert ImageProxy to Bitmap
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try {
            byte[] nv21 = yuv420ToNV21(image);
            YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21,
                    image.getWidth(), image.getHeight(), null);

            yuvImage.compressToJpeg(new Rect(0, 0,
                    image.getWidth(), image.getHeight()), 90, out);

            byte[] imageBytes = out.toByteArray();
            return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        } catch (Exception e) {
            Log.e(TAG, "Error converting image to bitmap", e);
            return null;
        } finally {
            try {
                out.close();
            } catch (IOException e) {
                Log.e(TAG, "Error closing output stream", e);
            }
        }
    }

    private byte[] yuv420ToNV21(ImageProxy image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width * height;
        int uvSize = width * height / 4;

        byte[] nv21 = new byte[ySize + uvSize * 2];

        // Get the YUV planes
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

        int rowStride = image.getPlanes()[0].getRowStride();
        int pixelStride = image.getPlanes()[1].getPixelStride();

        // Copy Y plane
        if (rowStride == width) {
            yBuffer.get(nv21, 0, ySize);
        } else {
            for (int row = 0; row < height; row++) {
                yBuffer.position(row * rowStride);
                yBuffer.get(nv21, row * width, width);
            }
        }

        // Interleave U and V planes
        int uvPosition = ySize;
        for (int row = 0; row < height / 2; row++) {
            for (int col = 0; col < width / 2; col++) {
                nv21[uvPosition++] = vBuffer.get(row * pixelStride * (width / 2) + col * pixelStride);
                nv21[uvPosition++] = uBuffer.get(row * pixelStride * (width / 2) + col * pixelStride);
            }
        }

        return nv21;
    }
}