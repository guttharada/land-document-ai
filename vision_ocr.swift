// Offline OCR with Apple Vision.  Build: swiftc -O vision_ocr.swift -o vision_ocr
// Usage: vision_ocr <list.txt> <out.jsonl> <tile px, 0 = whole image> <overlap px> <scale>
// Quads are written in full-image pixels, origin top-left.
import CoreGraphics
import Foundation
import ImageIO
import Vision

let a = CommandLine.arguments
let paths = try String(contentsOfFile: a[1], encoding: .utf8).split(separator: "\n").map(String.init)
let tile = Int(a[3])!, overlap = Int(a[4])!, scale = CGFloat(Double(a[5])!)
FileManager.default.createFile(atPath: a[2], contents: nil)
let out = FileHandle(forWritingAtPath: a[2])!

func load(_ p: String) -> CGImage? {
    guard let s = CGImageSourceCreateWithURL(URL(fileURLWithPath: p) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(s, 0, nil)
}

func resize(_ im: CGImage, _ k: CGFloat) -> CGImage {
    if k == 1 { return im }
    let w = Int(CGFloat(im.width) * k), h = Int(CGFloat(im.height) * k)
    let c = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                      space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    c.interpolationQuality = .high
    c.draw(im, in: CGRect(x: 0, y: 0, width: w, height: h))
    return c.makeImage()!
}

func ocr(_ im: CGImage, _ ox: CGFloat, _ oy: CGFloat) throws -> [[String: Any]] {
    let r = VNRecognizeTextRequest()
    r.recognitionLevel = .accurate
    r.recognitionLanguages = (ProcessInfo.processInfo.environment["OCR_LANGS"] ?? "th-TH,en-US").split(separator: ",").map(String.init)
    r.usesLanguageCorrection = false
    r.minimumTextHeight = 0  // do not skip small text
    try VNImageRequestHandler(cgImage: im, options: [:]).perform([r])
    let w = CGFloat(im.width), h = CGFloat(im.height)
    return (r.results ?? []).compactMap { o in
        guard let t = o.topCandidates(1).first else { return nil }
        let q = [o.topLeft, o.topRight, o.bottomRight, o.bottomLeft].map {
            [Double(($0.x * w) / scale + ox), Double(((1 - $0.y) * h) / scale + oy)]
        }
        return ["text": t.string, "conf": Double(t.confidence), "quad": q]
    }
}

for p in paths {
    try autoreleasepool {
        var obs: [[String: Any]] = []
        var err: Any = NSNull()
        let t0 = Date()
        if let im = load(p) {
            if tile == 0 {
                obs = try ocr(resize(im, scale), 0, 0)
            } else {
                let step = tile - overlap
                var y = 0
                while true {
                    var x = 0
                    while true {
                        let r = CGRect(x: x, y: y, width: min(tile, im.width - x), height: min(tile, im.height - y))
                        let found = try ocr(resize(im.cropping(to: r)!, scale), CGFloat(x), CGFloat(y))
                        // text cut by an inner tile edge is whole in the neighbouring tile
                        obs += found.filter { o in
                            let q = o["quad"] as! [[Double]]
                            let xs = q.map { $0[0] }, ys = q.map { $0[1] }, m = 4.0
                            return !((x > 0 && xs.min()! < Double(x) + m)
                                || (y > 0 && ys.min()! < Double(y) + m)
                                || (x + tile < im.width && xs.max()! > Double(x + tile) - m)
                                || (y + tile < im.height && ys.max()! > Double(y + tile) - m))
                        }
                        if x + tile >= im.width { break }
                        x += step
                    }
                    if y + tile >= im.height { break }
                    y += step
                }
            }
        } else { err = "cannot read image" }
        let rec: [String: Any] = ["image": p, "obs": obs, "error": err, "sec": Date().timeIntervalSince(t0)]
        out.write(try JSONSerialization.data(withJSONObject: rec))
        out.write("\n".data(using: .utf8)!)
    }
}
