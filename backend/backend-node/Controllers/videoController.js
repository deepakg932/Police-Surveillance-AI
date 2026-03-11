const axios = require("axios");
const Detection = require("../models/Detection.js");
const path = require("path");

// 1. SEARCH API: Purana data dhoondne ke liye
exports.searchDetections = async (req, res) => {
  try {
    const { object, textNote, trackingId, fileName } = req.query;
    let filter = {};

    // Filter Setup (Regex for dynamic search)
    if (object) filter.object = { $regex: object, $options: "i" };
    if (textNote) filter.textNote = { $regex: textNote, $options: "i" };
    if (trackingId) filter.trackingId = trackingId;
    if (fileName) filter.fileName = fileName;

    const results = await Detection.find(filter).sort({ createdAt: -1 });

    if (results.length === 0) {
      return res.json({
        message: "No results found",
        counts: {},
        totalUniqueObjects: 0,
        results: [],
      });
    }

    const finalCounts = {};
    const searchTerms = (object || results[0].textNote || "")
      .toLowerCase()
      .split(/[\s,]+/)
      .filter(
        (w) => !["with", "and", "wearing", "in", "a", "wear"].includes(w),
      );

    searchTerms.forEach((key) => {
      const uniqueSetForTerm = new Set();
      results.forEach((d) => {
        if (d.object.toLowerCase().includes(key)) {
          uniqueSetForTerm.add(d.trackingId);
        }
      });
      finalCounts[`total_${key}`] = uniqueSetForTerm.size;
    });

    return res.json({
      message: "Success",
      mode: "Search Result",
      counts: finalCounts,
      totalUniqueObjects: new Set(results.map((d) => d.trackingId)).size,
      results: results.map((d) => ({
        object: d.object,
        confidence: d.confidence,
        trackingId: d.trackingId,
        timestamp: d.timestamp,
        image_path: d.imagePath,
        bbox: d.bbox,
        processing_time: d.processingTime,
        screenshotUrl: d.screenshotUrl,
      })),
    });
  } catch (err) {
    console.error("❌ Search Error:", err.message);
    return res.status(500).json({ error: err.message });
  }
};

// 2. UPLOAD & PROCESS API: Naya video process karne ke liye
exports.uploadAndProcess = async (req, res) => {
  try {
    const videoFile = req.files?.file?.[0];
    const imageFile = req.files?.image?.[0]; // Optional image search
    const userPrompt = req.body.text || "person";

    if (!videoFile) return res.status(400).json({ message: "Video missing" });

    const startTime = Date.now();
    const BASE_URL = process.env.BASE_URL;

    // Video and Image URLs construct karna (Static serve ke liye)
    const videoUrl = `${BASE_URL}/uploads/${videoFile.filename}`;
    const imageUrl = imageFile
      ? `${BASE_URL}/uploads/${imageFile.filename}`
      : null;

    console.log("📤 Sending to Python API:", { videoUrl, userPrompt });

    // Python API Call (YOLO-World logic)
    const response = await axios.post(
      `${process.env.PYTHON_API_URL}/process`,
      {
        fileUrl: videoUrl,
        imageUrl: imageUrl,
        prompt: userPrompt,
      },
      { timeout: 0 }, // Large videos ke liye timeout hataya
    );

    const rawResults = response.data.results || [];

    // Filter: High confidence detections only
    const detections = rawResults.filter((d) => d.confidence >= 0.2);
    console.log(detections, "jjjjjjjjjjjjjrrrrr");

    const finalCounts = {};
    const keywords = userPrompt
      .toLowerCase()
      .split(/[\s,]+/)
      .filter((w) => !["with", "and", "a", "wearing"].includes(w));

    // Count unique objects per keyword
    keywords.forEach((key) => {
      const uniqueIds = new Set(
        detections
          .filter((d) => d.object.toLowerCase().includes(key))
          .map((d) => d.trackingId),
      );
      finalCounts[`total_${key}`] = uniqueIds.size;
    });

    // Database mein data save karna
    if (detections.length > 0) {
      const durationSeconds = ((Date.now() - startTime) / 1000).toFixed(2);

      const dbEntries = detections.map((d) => {
        // Safety Check: Agar image_path missing ho
        const rawPath = d.image_path || "detected_frames/default.jpg";
        const cleanPath = rawPath.replace(/\\/g, "/");

        return {
          fileName: videoFile.filename,
          textNote: userPrompt,
          object: d.object || "unknown",
          confidence: d.confidence,
          timestamp: d.timestamp,
          trackingId: d.trackingId ? d.trackingId.toString() : "0",
          bbox: d.bbox || [],
          imagePath: cleanPath,
          screenshotUrl: `${BASE_URL}/${cleanPath}`,
          processingTime: `${durationSeconds}s`,
        };
      });

      await Detection.insertMany(dbEntries);
      console.log(`✅ Saved ${dbEntries.length} detections to MongoDB`);
    } else {
      console.log("⚠️ No detections found with confidence >= 0.40");
    }

    // --- FINAL RESPONSE FIX ---
    // Alag variable mein map karke safety check lagaya hai
    const resultsForFrontend = detections.map((d) => {
      const safePath = (d.image_path || "detected_frames/default.jpg").replace(
        /\\/g,
        "/",
      );
      return {
        ...d,
        screenshotUrl: `${BASE_URL}/${safePath}`,
      };
    });

    console.log("🚀 Sending Final Response to Frontend...");

    return res.json({
      message: "Success",
      processing_time: `${((Date.now() - startTime) / 1000).toFixed(2)}s`,
      mode: imageFile ? "Image Search" : "Text Search",
      counts: finalCounts,
      totalUniqueObjects: new Set(detections.map((d) => d.trackingId)).size,
      results: resultsForFrontend, // Safe mapping use ki
    });
  } catch (err) {
    console.error("❌ ERROR in uploadAndProcess:", err.message);
    // User ko error response dena zaroori hai warna frontend 'loading' mein atka rahega
    if (!res.headersSent) {
      return res
        .status(500)
        .json({ error: "Processing failed", detail: err.message });
    }
  }
};
