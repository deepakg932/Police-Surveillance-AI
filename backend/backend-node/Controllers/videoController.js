const axios = require("axios");
const Detection = require("../models/Detection.js");

exports.uploadAndProcess = async (req, res) => {
  try {
    const videoFile = req.files?.file?.[0];
    const imageFile = req.files?.image?.[0];
    const userPrompt = req.body.text || "person";

    if (!videoFile) {
      return res.status(400).json({ message: "Video file missing" });
    }

    const startTime = Date.now();

    const BASE_URL = process.env.BASE_URL;

    const videoUrl = `${BASE_URL}/uploads/${videoFile.filename}`;
    const imageUrl = imageFile
      ? `${BASE_URL}/uploads/${imageFile.filename}`
      : null;

    let response;

    try {
      response = await axios.post(
        `${process.env.PYTHON_API_URL}/process`,
        {
          fileUrl: videoUrl,
          imageUrl: imageUrl,
          prompt: userPrompt,
        },
        {
          timeout: 7200000, // 2 hours for long videos
          maxContentLength: Infinity,
          maxBodyLength: Infinity,
        }
      );
    } catch (pythonErr) {
      console.error("❌ Python API Error:", pythonErr.message);

      return res.status(500).json({
        error: "Python AI processing failed",
        details: pythonErr.message,
      });
    }

    const endTime = Date.now();
    const processingTimeStr = `${((endTime - startTime) / 1000).toFixed(2)}s`;

    // Normalize Python response
    const detections = (response.data.results || [])
      .map((d, index) => {
        const cleanPath = (d.image_path || "").replace(/\\/g, "/");

        return {
          object: d.object || d.prompt || "unknown",
          ocrText: d.ocr_text || "",
          confidence:
            typeof d.confidence === "number" ? d.confidence : 0.85,
          trackingId: (
            d.trackingId || `track_${Date.now()}_${index}`
          ).toString(),
          timestamp: d.timestamp || "0",
          bbox: d.bbox || [],
          imagePath: cleanPath,
          color: d.color || null,
        };
      })
      .filter((d) => d.confidence >= 0.3);

    // Dynamic Counting
    const finalCounts = {};

    const normalizedPrompt = userPrompt
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, "")
      .trim();

    const uniqueSet = new Set();

    detections.forEach((d) => {
      const obj = (d.object || "").toLowerCase();
      const ocr = (d.ocrText || "").toLowerCase();

      if (obj.includes(normalizedPrompt) || ocr.includes(normalizedPrompt)) {
        uniqueSet.add(d.trackingId);
      }
    });

    finalCounts[`total_${normalizedPrompt.replace(/\s+/g, "_")}`] =
      uniqueSet.size;

    // Save to MongoDB
    if (detections.length > 0) {
      const detectionsToSave = detections.map((d) => {
        return {
          fileName: videoFile.filename,
          textNote: userPrompt,
          object: d.object,
          ocrText: d.ocrText,
          confidence: d.confidence,
          timestamp: d.timestamp,
          trackingId: d.trackingId,
          bbox: d.bbox,
          imagePath: d.imagePath,
          screenshotUrl: d.imagePath ? `${BASE_URL}/${d.imagePath}` : "",
          processingTime: processingTimeStr,
          videoUrl: videoUrl,
        };
      });

      await Detection.insertMany(detectionsToSave);
    }

    // Final Response
    return res.json({
      message: "Success",
      processing_time: processingTimeStr,
      mode: imageFile ? "Image Search" : "Text Search",
      counts: finalCounts,
      totalUniqueObjects: new Set(detections.map((d) => d.trackingId)).size,
      results: detections.map((d) => ({
        object: d.object,
        ocrText: d.ocrText,
        confidence: d.confidence,
        trackingId: d.trackingId,
        timestamp: d.timestamp,
        bbox: d.bbox,
        image_path: d.imagePath,
        screenshotUrl: d.imagePath ? `${BASE_URL}/${d.imagePath}` : "",
      })),
    });
  } catch (err) {
    console.error("❌ Node Error:", err);

    return res.status(500).json({
      error: err.message,
    });
  }
};

exports.searchDetections = async (req, res) => {
  try {
    const { object, textNote, trackingId, fileName } = req.query;

    let match = {};

    if (object) match.object = { $regex: object, $options: "i" };
    if (textNote) match.textNote = { $regex: textNote, $options: "i" };
    if (trackingId) match.trackingId = trackingId;
    if (fileName) match.fileName = fileName;

    const results = await Detection.aggregate([
      { $match: match },
      { $sort: { createdAt: -1 } },
      {
        $group: {
          _id: "$trackingId",
          doc: { $first: "$$ROOT" },
        },
      },
      { $replaceRoot: { newRoot: "$doc" } },
    ]);

    if (!results.length) {
      return res.json({
        message: "No results found",
        counts: {},
        totalUniqueObjects: 0,
        results: [],
      });
    }

    const searchPrompt = (object || textNote || "")
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, "")
      .trim();

    const finalCounts = {};
    const uniqueSet = new Set();

    results.forEach((d) => {
      const obj = (d.object || "").toLowerCase();
      const ocr = (d.ocrText || "").toLowerCase();

      if (obj.includes(searchPrompt) || ocr.includes(searchPrompt)) {
        uniqueSet.add(d.trackingId);
      }
    });

    finalCounts[`total_${searchPrompt.replace(/\s+/g, "_")}`] =
      uniqueSet.size;

    return res.json({
      message: "Success",
      mode: "Search Result",
      counts: finalCounts,
      totalUniqueObjects: results.length,
      results: results.map((d) => ({
        object: d.object,
        ocrText: d.ocrText,
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

    return res.status(500).json({
      error: err.message,
    });
  }
};