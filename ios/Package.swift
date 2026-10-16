// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "PodcastAdTagger",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "PodcastAdTagger", targets: ["PodcastAdTagger"]),
        .executable(name: "adtag", targets: ["adtag"]),
    ],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", exact: "1.24.2"),
    ],
    targets: [
        // llama.cpp's own prebuilt xcframework (iOS, iOS simulator, macOS), used for the embedder.
        .binaryTarget(
            name: "llama",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-xcframework.zip",
            checksum: "937891f26734c29c80817cdf8bf59b139c84a9c6fb36a4ad6ce31cceea608e7c"),
        .target(
            name: "PodcastAdTagger",
            dependencies: [
                "llama",
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ]),
        .executableTarget(name: "adtag", dependencies: ["PodcastAdTagger"]),
        .testTarget(name: "PodcastAdTaggerTests", dependencies: ["PodcastAdTagger"]),
    ]
)
