// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ABI",
    platforms: [.iOS(.v15), .macOS(.v13)],
    products: [
        .library(name: "ABI", targets: ["ABI"]),
    ],
    targets: [
        .systemLibrary(name: "CABI", path: "Sources/CABI"),
        .target(name: "ABI", dependencies: ["CABI"], path: "Sources/ABI"),
    ]
)
