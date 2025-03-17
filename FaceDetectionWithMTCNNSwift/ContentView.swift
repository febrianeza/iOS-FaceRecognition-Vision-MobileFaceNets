//
//  ContentView.swift
//  FaceDetectionWithMTCNNSwift
//
//  Created by EzaFebrian on 11/03/25.
//

import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var viewModel: ContentViewModel = ContentViewModel()
    var body: some View {
        VStack {
            Text(viewModel.action)
                .font(.system(size: 14).bold())
                .padding(.bottom, 16)
            
            CameraPreview(session: cameraManager.session)
                .frame(width: min(UIScreen.main.bounds.width, 270),
                       height: min(UIScreen.main.bounds.width, 270))
                .aspectRatio(1.0, contentMode: .fit) // Enforce 1:1 ratio
                .clipShape(Circle()) // Clip to ensure square bounds
            
            Spacer()
            
            Text("Result: \(viewModel.result)")
            
            Spacer()
            
            ZStack {
                // Capture Button
                viewModel.isLoading
                ? AnyView(ProgressView())
                : AnyView(Button(action: {
                    if !viewModel.isLoading {
                        viewModel.isLoading = true
                        cameraManager.capturePhoto()
                    }
                }) {
                    Circle()
                        .frame(width: 70, height: 70)
                        .foregroundColor(.white)
                        .overlay(
                            Circle()
                                .frame(width: 60, height: 60)
                                .foregroundColor(.gray)
                        )
                })
            }
            .frame(height: 100)
            .padding(.bottom, 30)
        }
        .onAppear {
            cameraManager.setupCamera()
            
            cameraManager.captured = { image in
                if let data = image {
                    viewModel.detectFace(image: data)
                }
            }
        }
        .padding()
    }
}

// Camera Manager Class
class CameraManager: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate {
    let session = AVCaptureSession()
    private var photoOutput = AVCapturePhotoOutput()
    private let sessionQueue = DispatchQueue(label: "session queue")
    var captured: ((UIImage?) -> ())?
    
    func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .photo
        
        // Setup camera input
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                        for: .video,
                                                        position: .front),
              let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice),
              session.canAddInput(videoDeviceInput) else {
            return
        }

        session.addInput(videoDeviceInput)
        
        // Setup photo output
        if session.canAddOutput(photoOutput) {
            session.addOutput(photoOutput)
            let activeFormat = videoDevice.activeFormat
            let supportedDimensions = activeFormat.supportedMaxPhotoDimensions
            for dimen in supportedDimensions {
                print("supported: width: \(dimen.width), height: \(dimen.height)")
            }
            if let firstSupportedDimension = supportedDimensions.first {
                photoOutput.maxPhotoDimensions = firstSupportedDimension
                print("Set maxPhotoDimensions to: \(firstSupportedDimension.width)x\(firstSupportedDimension.height)")
            }
        }
        
        session.commitConfiguration()
        
        // Start session on a background thread
        sessionQueue.async {
            self.session.startRunning()
        }
    }
    
    func capturePhoto() {
        let settings = AVCapturePhotoSettings()
        
        let maxQuality = photoOutput.maxPhotoQualityPrioritization
        settings.photoQualityPrioritization = maxQuality
        
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard let imageData = photo.fileDataRepresentation() else {
            print("Failed to get image data")
            return
        }
        
        guard let capturedImage = UIImage(data: imageData ) else { return }
        
        // Resize the image to
        guard let resizedImageData = capturedImage.resizeImageToPixelWidth(newPixelWidth: CGFloat(720)) else {
            print("Failed to convert resized image to data")
            return
        }
        
        self.captured?(resizedImageData)
    }
}

// Camera Preview UIViewControllerRepresentable
struct CameraPreview: UIViewControllerRepresentable {
    let session: AVCaptureSession
    
    func makeUIViewController(context: Context) -> UIViewController {
        let viewController = UIViewController()
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        
        // Ensure square preview
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = CGRect(x: 0, y: 0, width: 270, height: 270)
        
        viewController.view.layer.addSublayer(previewLayer)
        viewController.view.clipsToBounds = true
        
        return viewController
    }
    
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        // Update preview layer frame if needed
        if let previewLayer = uiViewController.view.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            previewLayer.frame = CGRect(x: 0, y: 0, width: 270, height: 270)
        }
    }
}

#Preview {
    ContentView()
}

extension UIImage {
    func resizeImageToPixelWidth(newPixelWidth: CGFloat) -> UIImage? {
        // Calculate original pixel dimensions
        let originalPixelWidth = self.size.width * self.scale
        let originalPixelHeight = self.size.height * self.scale
        
        // Calculate aspect ratio and new pixel height
        let aspectRatio = originalPixelHeight / originalPixelWidth
        let newPixelHeight = newPixelWidth * aspectRatio
        
        // Define the new size in points (with scale 1.0, points = pixels)
        let newSize = CGSize(width: newPixelWidth, height: newPixelHeight)
        
        // Configure the renderer format with scale 1.0
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        
        // Create the renderer and generate the new image
        let renderer = UIGraphicsImageRenderer(size: newSize, format: format)
        let newImage = renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: newSize))
        }
        
        return newImage
    }
}
