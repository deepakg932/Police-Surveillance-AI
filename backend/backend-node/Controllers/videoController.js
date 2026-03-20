const axios = require("axios");
const Detection = require("../models/Detection.js");
const mongoose = require("mongoose")


// exports.searchDetections = async (req, res) => {
//   try {
//       const userId = req.userId; // ✅ middleware se aaya
//     const { object, textNote, trackingId, fileName } = req.query;

//     let match = {};

//     if (object) match.object = { $regex: object, $options: "i" };
//     if (textNote) match.textNote = { $regex: textNote, $options: "i" };
//     if (trackingId) match.trackingId = trackingId;
//     if (fileName) match.fileName = fileName;

//     const results = await Detection.aggregate([
//       { $match: match },
//       { $sort: { createdAt: -1 } },
//       {
//         $group: {
//           _id: "$trackingId",
//           doc: { $first: "$$ROOT" },
//         },
//       },
//       { $replaceRoot: { newRoot: "$doc" } },
//     ]);

//     if (!results.length) {
//       return res.json({
//         message: "No results found",
//         counts: {},
//         totalUniqueObjects: 0,
//         results: [],
//       });
//     }

//     const searchPrompt = (object || textNote || "")
//       .toLowerCase()
//       .replace(/[^a-z0-9\s]/g, "")
//       .trim();

//     const finalCounts = {};
//     const uniqueSet = new Set();

//     results.forEach((d) => {
//       const obj = (d.object || "").toLowerCase();
//       const ocr = (d.ocrText || "").toLowerCase();

//       if (obj.includes(searchPrompt) || ocr.includes(searchPrompt)) {
//         uniqueSet.add(d.trackingId);
//       }
//     });

//     finalCounts[`total_${searchPrompt.replace(/\s+/g, "_")}`] = uniqueSet.size;

//     return res.json({
//       message: "Success",
//       mode: "Search Result",
//       counts: finalCounts,
//       totalUniqueObjects: results.length,
//       results: results.map((d) => ({
//         object: d.object,
//         ocrText: d.ocrText,
//         confidence: d.confidence,
//         trackingId: d.trackingId,
//         timestamp: d.timestamp,
//         image_path: d.imagePath,
//         bbox: d.bbox,
//         processing_time: d.processingTime,
//         screenshotUrl: d.screenshotUrl,
//       })),
//     });
//   } catch (err) {
//     console.error("❌ Search Error:", err.message);

//     return res.status(500).json({
//       error: err.message,
//     });
//   }
// };



exports.searchDetections = async (req, res) => {
  try {
    const userId = req.userId;
    const { object, textNote, trackingId, fileName } = req.query;

    // if (!userId) {
    //   return res.status(401).json({ message: "Unauthorized" });
    // }

    // ✅ FIXED
    let match = {
      userId: new mongoose.Types.ObjectId(userId),
    };

    if (object) {
      match.object = { $regex: object, $options: "i" };
    }

    if (textNote) {
      match.textNote = { $regex: textNote, $options: "i" };
    }

    if (trackingId) {
      match.trackingId = { $regex: trackingId, $options: "i" };
    }

    if (fileName) {
      match.fileName = { $regex: fileName, $options: "i" };
    }

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

    const uniqueSet = new Set();

    results.forEach((d) => {
      const obj = (d.object || "").toLowerCase();
      if (!searchPrompt || obj.includes(searchPrompt)) {
        uniqueSet.add(d.trackingId);
      }
    });

    return res.json({
      message: "Success",
      counts: searchPrompt
        ? {
            [`total_${searchPrompt.replace(/\s+/g, "_")}`]:
              uniqueSet.size,
          }
        : {},
      totalUniqueObjects: results.length,
      results: results.map((d) => ({
        object: d.object,
        ocrText: d.ocrText,
        confidence: d.confidence,
        trackingId: d.trackingId,
        timestamp: d.timestamp,
        image_path: d.imagePath,
        bbox: d.bbox,
        screenshotUrl: d.screenshotUrl,
      })),
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.message });
  }
};

// exports.uploadAndProcess = async (req, res) => {
//   try {
//     const userId = req.userId; // ✅ middleware se aaya
//     const videoFile = req.files?.file?.[0];
//     console.log(videoFile, "kkk");
//     const imageFile = req.files?.image?.[0];
//     const userPrompt = req.body.text || "person";
//     console.log(userPrompt, "okk");

//     if (!videoFile) {
//       return res.status(400).json({ message: "Video file missing" });
//     }

//     const startTime = Date.now();

//     const BASE_URL = process.env.BASE_URL;

//     const videoUrl = `${BASE_URL}/uploads/${videoFile.filename}`;
//     const imageUrl = imageFile
//       ? `${BASE_URL}/uploads/${imageFile.filename}`
//       : null;

//     let response;

//     try {
//       response = await axios.post(
//         `${process.env.PYTHON_API_URL}/process`,
//         {
//           fileUrl: videoUrl,
//           imageUrl: imageUrl,
//           prompt: userPrompt,
//         },
//         {
//           timeout: 7200000, // 2 hours for long videos
//           maxContentLength: Infinity,
//           maxBodyLength: Infinity,
//         },
//       );
//     } catch (pythonErr) {
//       console.error("❌ Python API Error:", pythonErr.message);

//       return res.status(500).json({
//         error: "Python AI processing failed",
//         details: pythonErr.message,
//       });
//     }

//     const endTime = Date.now();
//     const processingTimeStr = `${((endTime - startTime) / 1000).toFixed(2)}s`;

//     // Normalize Python response
//     const detections = (response.data.results || [])
//       .map((d, index) => {
//         const cleanPath = (d.image_path || "").replace(/\\/g, "/");

//         return {
//           object: d.object || d.prompt || "unknown",
//           ocrText: d.ocr_text || "",
//           confidence: typeof d.confidence === "number" ? d.confidence : 0.85,
//           trackingId: (
//             d.trackingId || `track_${Date.now()}_${index}`
//           ).toString(),
//           timestamp: d.timestamp || "0",
//           bbox: d.bbox || [],
//           imagePath: cleanPath,
//           color: d.color || null,
//         };
//       })
//       .filter((d) => d.confidence >= 0.3);

//     // Dynamic Counting
//     const finalCounts = {};

//     const normalizedPrompt = userPrompt
//       .toLowerCase()
//       .replace(/[^a-z0-9\s]/g, "")
//       .trim();

    

//     finalCounts[`total_${normalizedPrompt.replace(/\s+/g, "_")}`] =
//       uniqueSet.size;

//     // Save to MongoDB
//     if (detections.length > 0) {
//       const detectionsToSave = detections.map((d) => {
//         return {
//           fileName: videoFile.filename,
//           textNote: userPrompt,
//           object: d.object,
//           ocrText: d.ocrText,
//           confidence: d.confidence,
//           timestamp: d.timestamp,
//           trackingId: d.trackingId,
//           bbox: d.bbox,
//           imagePath: d.imagePath,
//           screenshotUrl: d.imagePath ? `${BASE_URL}/${d.imagePath}` : "",
//           processingTime: processingTimeStr,
//           videoUrl: videoUrl,
//         };
//       });



//       await Detection.insertMany(detectionsToSave);
//     }




//     const uniqueSet = new Set();

//     detections.forEach((d) => {
//       const obj = (d.object || "").toLowerCase();
//       const ocr = (d.ocrText || "").toLowerCase();

//       if (obj.includes(normalizedPrompt) || ocr.includes(normalizedPrompt)) {
//         uniqueSet.add(d.trackingId);
//       }
//     });
//     // Final Response
//     return res.json({
//       message: "Success",
//       processing_time: processingTimeStr,
//       mode: imageFile ? "Image Search" : "Text Search",
//       counts: finalCounts,
//       totalUniqueObjects: new Set(detections.map((d) => d.trackingId)).size,
//       results: detections.map((d) => ({
//         object: d.object,
//         ocrText: d.ocrText,
//         confidence: d.confidence,
//         trackingId: d.trackingId,
//         timestamp: d.timestamp,
//         bbox: d.bbox,
//         image_path: d.imagePath,
//         screenshotUrl: d.imagePath ? `${BASE_URL}/${d.imagePath}` : "",
//       })),
//     });
//   } catch (err) {
//     console.error("❌ Node Error:", err);

//     return res.status(500).json({
//       error: err.message,
//     });
//   }
// };



exports.uploadAndProcess = async (req, res) => {
  try {
    const userId = req.userId;
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
        { fileUrl: videoUrl, imageUrl, prompt: userPrompt },
        {
          timeout: 7200000,
          maxContentLength: Infinity,
          maxBodyLength: Infinity,
        }
      );
    } catch (pythonErr) {
      return res.status(500).json({
        error: "Python AI processing failed",
        details: pythonErr.message,
      });
    }

    const endTime = Date.now();
    const processingTimeStr = `${((endTime - startTime) / 1000).toFixed(2)}s`;

    // ✅ DEBUG — dekho Python se kya aa raha hai
    console.log("=== PYTHON RESPONSE ===");
    console.log("Total from Python:", response.data.results?.length);
    console.log("Mode:", response.data.mode);
    console.log("Sample:", JSON.stringify(response.data.results?.slice(0, 2)));
    console.log("======================");

    const detections = (response.data.results || []).map((d, index) => {
      // ✅ image_path clean karo
      const cleanPath = (d.image_path || "").replace(/\\/g, "/");

      // ✅ confidence — mode ke hisaab se
      let conf = 0.85;
      if (typeof d.confidence === "number") conf = d.confidence;
      else if (d.plate)                     conf = 0.90;
      else if (typeof d.age === "number")   conf = 0.85;

      // ✅ bbox — multiple fields check karo
      let bbox = [];
      if (Array.isArray(d.bbox) && d.bbox.length === 4) {
        bbox = d.bbox;
      } else if (Array.isArray(d.plate_bbox) && d.plate_bbox.length === 4) {
        bbox = d.plate_bbox;
      } else if (Array.isArray(d.person_bbox) && d.person_bbox.length === 4) {
        bbox = d.person_bbox;
      } else if (Array.isArray(d.vehicle_bbox) && d.vehicle_bbox.length === 4) {
        bbox = d.vehicle_bbox;
      }

      // ✅ ocrText — multiple fields check karo
      const ocrText = d.ocr_text || d.plate || d.ocrText || "";

      // ✅ object label
      const objectLabel = d.object || d.prompt || "unknown";

      // ✅ color
      const color = d.color || d.vehicle || "";

      return {
        object:     objectLabel,
        ocrText:    ocrText,
        confidence: conf,
        trackingId: (
          d.trackingId || `track_${Date.now()}_${index}`
        ).toString(),
        timestamp:  typeof d.timestamp === "number" ? d.timestamp : 0,
        bbox:       bbox,
        imagePath:  cleanPath,
        color:      color,
      };
    });

    console.log("After mapping:", detections.length);

    // ✅ Save to MongoDB
    if (detections.length > 0) {
      const detectionsToSave = detections.map((d) => {
        // ✅ bbox extra sanitize
        let safeBbox = [];
        if (Array.isArray(d.bbox) && d.bbox.length === 4) {
          safeBbox = d.bbox.map(Number).filter((n) => !isNaN(n));
          if (safeBbox.length !== 4) safeBbox = [];
        }

        return {
          userId:         new mongoose.Types.ObjectId(userId),
          fileName:       videoFile.filename,
          textNote:       userPrompt,
          object:         d.object,
          ocrText:        d.ocrText,
          confidence:     d.confidence,
          timestamp:      d.timestamp,
          trackingId:     d.trackingId,
          bbox:           safeBbox,
          imagePath:      d.imagePath,
          screenshotUrl:  d.imagePath ? `${BASE_URL}/${d.imagePath}` : "",
          processingTime: processingTimeStr,
          videoUrl:       videoUrl,
          color:          d.color || "",
        };
      });

      try {
        await Detection.insertMany(detectionsToSave, { ordered: false });
        console.log("Saved to DB:", detectionsToSave.length);
      } catch (dbErr) {
        console.error("DB Save Error:", dbErr.message);
      }
    }

    // ✅ Counting
    const uniqueTrackingIds = new Set(detections.map((d) => d.trackingId));
    const normalizedPrompt = userPrompt
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, "")
      .trim();

    return res.json({
      message:          "Success",
      processing_time:  processingTimeStr,
      mode:             response.data.mode || "unknown",
      total_found:      detections.length,
      totalUniqueObjects: uniqueTrackingIds.size,
      counts: {
        [`total_${normalizedPrompt.replace(/\s+/g, "_")}`]: detections.length,
      },
      results: detections.map((d) => ({
        object:       d.object,
        ocrText:      d.ocrText,
        confidence:   d.confidence,
        trackingId:   d.trackingId,
        timestamp:    d.timestamp,
        bbox:         d.bbox,
        image_path:   d.imagePath,
        screenshotUrl: d.imagePath ? `${BASE_URL}/${d.imagePath}` : "",
        color:        d.color,
      })),
    });
  } catch (err) {
    console.error("❌ uploadAndProcess Error:", err);
    return res.status(500).json({ error: err.message });
  }
};


exports.searchDetections = async (req, res) => {
  try {
    const userId = req.userId;
    const { object, textNote, trackingId, fileName } = req.query;

    if (!userId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    let match = {
      userId: new mongoose.Types.ObjectId(userId),
    };

    if (object)     match.object     = { $regex: object,     $options: "i" };
    if (textNote)   match.textNote   = { $regex: textNote,   $options: "i" };
    if (trackingId) match.trackingId = { $regex: trackingId, $options: "i" };
    if (fileName)   match.fileName   = { $regex: fileName,   $options: "i" };

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
      { $sort: { createdAt: -1 } },
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

    return res.json({
      message: "Success",
      totalUniqueObjects: results.length,
      counts: searchPrompt
        ? {
            [`total_${searchPrompt.replace(/\s+/g, "_")}`]: results.length,
          }
        : { total: results.length },
      results: results.map((d) => ({
        object:       d.object,
        ocrText:      d.ocrText,
        confidence:   d.confidence,
        trackingId:   d.trackingId,
        timestamp:    d.timestamp,
        image_path:   d.imagePath,
        bbox:         d.bbox,
        color:        d.color,
        screenshotUrl: d.screenshotUrl,
        textNote:     d.textNote,
        fileName:     d.fileName,
        createdAt:    d.createdAt,
      })),
    });
  } catch (err) {
    console.error("❌ searchDetections Error:", err);
    return res.status(500).json({ error: err.message });
  }
};