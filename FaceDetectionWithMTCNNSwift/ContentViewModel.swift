//
//  ContentViewModel.swift
//  FaceDetectionWithMTCNNSwift
//
//  Created by EzaFebrian on 12/03/25.
//

import Foundation
import UIKit
import Vision
import SwiftUI
import TensorFlowLite

enum State {
    case comparing
    case registering
}

class ContentViewModel: ObservableObject {
    
    @Published var isLoading: Bool = false
    @Published var action: String = "Registering"
    @Published var result: String = ""
    
    private var interpreter: Interpreter?
    var state: State = .registering
    
    init() {
        setupModel()
    }
    
    func detectFace(image: UIImage) {
        guard let cgImage = image.cgImage else { return }
        
        let request = VNImageRequestHandler(cgImage: cgImage, orientation: .up)
        let detectFaceRequest = VNDetectFaceRectanglesRequest { (request, error) in
            Task { 
                await self.handleFaceDetection(request: request, error: error, inputImage: image) }
        }
        try? request.perform([detectFaceRequest])
    }
    
    private func handleFaceDetection(request: VNRequest, error: Error?, inputImage: UIImage?) async {
        DispatchQueue.main.async { [weak self] in
            self?.isLoading = true
        }
        
        guard let observations = request.results as? [VNFaceObservation],
              let originalImage = inputImage else { return }
        
        // if a face detected.
        if !observations.isEmpty {
            let observation = observations[0]
            let boundingBox = observation.boundingBox
            
            // Convert normalized bounding box to pixel coordinates
            let imageSize = originalImage.size
            let x = boundingBox.origin.x * imageSize.width
            let height = boundingBox.size.height * imageSize.height
            let y = (1 - boundingBox.origin.y - boundingBox.size.height) * imageSize.height
            let width = boundingBox.size.width * imageSize.width
            
            let croppingRect = CGRect(x: x, y: y, width: width, height: height)
            
            if let imageData = originalImage.jpegData(compressionQuality: 0.8) {
                saveImageToDocumentDirectory(imageData: imageData, fileName: self.state == .registering ? "compressed_photo.jpg" : "comparing_photo.jpg")
            }
            
            // Crop the image to the bounding box
            if let croppedCGImage = originalImage.cgImage?.cropping(to: croppingRect) {
                let croppedImage = UIImage(cgImage: croppedCGImage)
                
                let landmarks = await self.getFaceLandmarks(from: croppedImage)
                if let landmarks = landmarks {
                    let faceAligned = await self.faceAlign(image: croppedImage, landmarks: landmarks)
                    
                    // Convert the cropped image to JPEG data and save it
                    if let jpegData = faceAligned.jpegData(compressionQuality: 0.8) {
                        saveImageToDocumentDirectory(imageData: jpegData, fileName: self.state == .registering ? "cropped_photo.jpg" : "comparing_cropped_photo.jpg")
                    }
                }
            }
        } else {print("Face is not detected")}
        
        DispatchQueue.main.async { [weak self] in
            if self?.state == .comparing {
                self?.compareFaces()
            }
            
            self?.isLoading = false
            self?.action = "Comparing"
            self?.state = .comparing
        }
    }
    
    func faceAlign(image: UIImage, landmarks: [CGPoint]) async -> UIImage {
        // Ensure we have at least 2 landmarks (left and right eye)
        guard landmarks.count >= 2 else { return image }
        
        // Extract eye positions
        let leftEye = landmarks[0]
        let rightEye = landmarks[1]
        
        // Calculate differences in coordinates
        let diffEyeX = rightEye.x - leftEye.x
        let diffEyeY = rightEye.y - leftEye.y
        
        // Calculate rotation angle in radians
        let alpha: CGFloat
        if abs(diffEyeY) < 1e-7 {
            alpha = 0
        } else {
            let ratio = diffEyeY / diffEyeX
            alpha = CGFloat(atan(Double(ratio)))
        }
        
        // Create a rotation transform (negative angle to align eyes horizontally)
        let rotationTransform = CGAffineTransform(rotationAngle: -alpha)
        
        // Get the image size
        let size = image.size
        
        // Create a new image context with the same size and scale
        UIGraphicsBeginImageContextWithOptions(size, false, image.scale)
        let context = UIGraphicsGetCurrentContext()!
        
        // Apply the rotation transform to the context
        context.concatenate(rotationTransform)
        
        // Draw the original image into the transformed context
        image.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        
        // Retrieve the new rotated image
        let newImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        return newImage
    }
    
    func getFaceLandmarks(from image: UIImage) async -> [CGPoint]? {
        // Convert UIImage to CIImage
        guard let ciImage = CIImage(image: image) else { return nil }
        
        // Set up the face landmark detection request
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        
        // Perform the detection
        try? handler.perform([request])
        
        // Ensure exactly one face is detected
        guard let observations = request.results, observations.count == 1 else {
            return nil
        }
        
        // Get the single face observation
        let faceObservation = observations.first!
        guard let landmarks = faceObservation.landmarks else { return nil }
        
        let imageSize = image.size
        
        // Helper function to compute the center of a landmark region
        func centerOfLandmark(_ landmark: VNFaceLandmarkRegion2D?) -> CGPoint? {
            guard let landmark = landmark, !landmark.normalizedPoints.isEmpty else { return nil }
            let points = landmark.normalizedPoints
            let sum = points.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
            let average = CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
            // Convert normalized coordinates to UIImage coordinates
            let x = average.x * imageSize.width
            let y = (1 - average.y) * imageSize.height // Flip y-axis
            return CGPoint(x: x, y: y)
        }
        
        // Compute the centers of the left and right eyes
        guard let leftEyeCenter = centerOfLandmark(landmarks.leftEye),
              let rightEyeCenter = centerOfLandmark(landmarks.rightEye) else {
            return nil
        }
        
        // Return the landmarks as an array
        return [leftEyeCenter, rightEyeCenter]
    }
    
    // MARK: - Face Comparison
    private func compareFaces() {
        guard let interpreter = interpreter else { return }
        
        // Load the two images
        guard let image1 = loadImageFromDocuments(fileName: "cropped_photo.jpg"),
              let image2 = loadImageFromDocuments(fileName: "comparing_cropped_photo.jpg") else {
                  print("Failed to load images.")
                  return
              }
        
        guard let resizedImage1 = image1.resize(to: CGSize(width: 112, height: 112)),
              let resizedImage2 = image2.resize(to: CGSize(width: 112, height: 112)) else {
            print("Failed to resize images.")
            return
        }
        
        // Get embeddings
        guard let (embeddings1, embeddings2) = getEmbeddings2(from: resizedImage1, and: resizedImage2, using: interpreter) else {
            print("Failed to get embeddings.")
            return
        }
        
        let distance = cosineSimilarity(embeddings1, embeddings2)
        print("Distance between embeddings: \(distance)")
        
        let threshold: Float = 0.75
        DispatchQueue.main.async { [weak self] in
            self?.result = distance >= threshold ? "Same person" : "Different persons"
        }
    }
    
    func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.infinity }
        return sqrt(zip(a, b).map { pow($0 - $1, 2) }.reduce(0, +))
    }
    
    func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        return zip(a, b).map(*).reduce(0, +)
    }
    
    func magnitude(_ a: [Float]) -> Float {
        return sqrt(a.map { $0 * $0 }.reduce(0, +))
    }
    
    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        let dot = dotProduct(a, b)
        let magA = magnitude(a)
        let magB = magnitude(b)
        guard magA != 0 && magB != 0 else { return 0 }
        return dot / (magA * magB)
    }
    
    private func imageToFloatArray(image: UIImage) -> [Float]? {
        // Ensure the image is valid
        guard let cgImage = image.cgImage else { return nil }
        let width = 112
        let height = 112
        
        // Check image dimensions
        guard cgImage.width == width, cgImage.height == height else {
            print("Image must be 112x112, got \(cgImage.width)x\(cgImage.height)")
            return nil
        }
        
        // Set up the bitmap context
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4 // ARGB
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        let bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        
        var data = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        guard let context = CGContext(data: &data, width: width, height: height,
                                     bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow,
                                     space: colorSpace, bitmapInfo: bitmapInfo) else {
            return nil
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Convert pixel data to normalized float values
        var floatArray = [Float]()
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * bytesPerPixel
                let r = Float(data[offset + 1]) // Red
                let g = Float(data[offset + 2]) // Green
                let b = Float(data[offset + 3]) // Blue
                // Normalize to [-1, 1] range (common for face recognition models)
                floatArray.append((r - 127.5) / 128.0)
                floatArray.append((g - 127.5) / 128.0)
                floatArray.append((b - 127.5) / 128.0)
            }
        }
        
        return floatArray // Size: 37,632 floats
    }
    
    private func getEmbeddings2(from image1: UIImage, and image2: UIImage, using interpreter: Interpreter?) -> ([Float], [Float])? {
        // Ensure interpreter is initialized
        guard let interpreter = interpreter else {
            print("Interpreter not initialized")
            return nil
        }
        
        // Convert both images to float arrays
        guard let floatData1 = imageToFloatArray(image: image1),
              let floatData2 = imageToFloatArray(image: image2) else {
            print("Failed to convert images to float arrays")
            return nil
        }
        
        // Combine into a single batch
        let batchFloatData = floatData1 + floatData2 // Size: 75,264 floats
        
        do {
            // Verify input tensor shape
            let inputTensor = try interpreter.input(at: 0)
            let expectedSize = inputTensor.shape.dimensions.reduce(1, *) // 75,264
            guard batchFloatData.count == expectedSize else {
                print("Data count mismatch: expected \(expectedSize), got \(batchFloatData.count)")
                return nil
            }
            
            // Copy data to the input tensor
            let inputData = Data(buffer: UnsafeBufferPointer(start: batchFloatData, count: batchFloatData.count))
            try interpreter.copy(inputData, toInputAt: 0)
            
            // Run the model
            try interpreter.invoke()
            
            // Retrieve output (e.g., embeddings for both images)
            let outputTensor = try interpreter.output(at: 0)
            let embeddings = [Float](unsafeData: outputTensor.data) ?? []
            
            // Split embeddings into two sets (assuming output shape is [2, embedding_size])
            let embeddingSize = embeddings.count / 2
            let embeddings1 = Array(embeddings[0..<embeddingSize])
            let embeddings2 = Array(embeddings[embeddingSize..<embeddings.count])
            return (embeddings1, embeddings2)
        } catch {
            print("Inference error: \(error)")
            return nil
        }
    }
    
    // MARK: - Model Setup
    private func setupModel() {
        guard let modelPath = Bundle.main.path(forResource: "MobileFaceNet", ofType: "tflite") else {
            print("Failed to find model file.")
            return
        }
        
        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter?.allocateTensors()
        } catch {
            print("Failed to initialize interpreter: \(error)")
        }
    }
    
    // MARK: - Image Preprocessing
    private func imageToPixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        let width = 112
        let height = 112
        
        // Create pixel buffer
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
             kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard status == kCVReturnSuccess, let unwrappedPixelBuffer = pixelBuffer else {
            return nil
        }
        
        // Draw image into pixel buffer
        CVPixelBufferLockBaseAddress(unwrappedPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(unwrappedPixelBuffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(unwrappedPixelBuffer),
                                      space: rgbColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            CVPixelBufferUnlockBaseAddress(unwrappedPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
            return nil
        }
        
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1.0, y: -1.0)
        UIGraphicsPushContext(context)
        image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(unwrappedPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        // Normalize pixel buffer (MobileFaceNet: subtract 127.5, divide by 128)
        guard let data = pixelBufferToFloatArray(unwrappedPixelBuffer) else { return nil }
        let normalizedData = data.map { ($0 - 127.5) / 128.0 }
        
        // Copy normalized data back to a new pixel buffer
        var normalizedPixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attrs, &normalizedPixelBuffer)
        guard let newBuffer = normalizedPixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(newBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let newPixelData = CVPixelBufferGetBaseAddress(newBuffer)
        memcpy(newPixelData, normalizedData, normalizedData.count * MemoryLayout<Float>.size)
        CVPixelBufferUnlockBaseAddress(newBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return newBuffer
    }
    
    // MARK: - Inference
    private func getEmbeddings(from pixelBuffer: CVPixelBuffer) -> [Float]? {
        guard let interpreter = interpreter else { return nil }
        do {
            // Prepare input tensor
            let inputTensor = try interpreter.input(at: 0)
            
            print("Input tensor shape: \(inputTensor.shape), data type: \(inputTensor.dataType)")
            
            // Convert pixel buffer to float array for input
            guard let floatData = pixelBufferToFloatArray(pixelBuffer) else { return nil }
            let inputData = Data(buffer: UnsafeBufferPointer(start: floatData, count: floatData.count))
            
            // Copy data to input tensor
            try interpreter.copy(inputData, toInputAt: 0)
            
            // Run inference
            try interpreter.invoke()
            
            // Get output tensor
            let outputTensor = try interpreter.output(at: 0)
            let embeddings = [Float](unsafeData: outputTensor.data) ?? []
            return embeddings
        } catch {
            print("Inference error: \(error)")
            return nil
        }
    }
    
    private func pixelBufferToFloatArray(_ pixelBuffer: CVPixelBuffer) -> [Float]? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            return nil
        }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var floatArray: [Float] = []
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                let r = Float(buffer[offset + 1]) // Red
                let g = Float(buffer[offset + 2]) // Green
                let b = Float(buffer[offset + 3]) // Blue
                floatArray.append(r)
                floatArray.append(g)
                floatArray.append(b)
            }
        }
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        return floatArray
    }
}

func saveImageToDocumentDirectory(imageData: Data, fileName: String) {
    let documentDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let fileURL = documentDirectory.appendingPathComponent(fileName)
    do {
        try imageData.write(to: fileURL)
        print("Image saved to \(fileURL)")
    } catch {
        print("Error saving image: \(error)")
    }
}

func loadImageFromDocuments(fileName: String) -> UIImage? {
    do {
        // Get document directory
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        
        // Create file URL with the provided name
        let fileURL = documentsURL.appendingPathComponent(fileName)
        
        // Check if file exists
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            print("File does not exist at path: \(fileURL.path)")
            return nil
        }
        
        // Read the data
        let imageData = try Data(contentsOf: fileURL)
        
        // Convert to UIImage
        guard let image = UIImage(data: imageData) else {
            print("Failed to create UIImage from data")
            return nil
        }
        
        return image
        
    } catch {
        print("Error loading image: \(error.localizedDescription)")
        return nil
    }
}

// Helper to resize UIImage
extension UIImage {
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}

// Ensure this extension is in your codebase
extension Array where Element == Float {
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Float>.stride == 0 else { return nil }
        let count = unsafeData.count / MemoryLayout<Float>.stride
        self = unsafeData.withUnsafeBytes { buffer in
            Array(UnsafeBufferPointer(start: buffer.baseAddress!.assumingMemoryBound(to: Float.self), count: count))
        }
    }
}
