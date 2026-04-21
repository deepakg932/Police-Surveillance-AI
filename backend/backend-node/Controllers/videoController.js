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

    const { jobId } = req.query;
    if (jobId) {
  match.jobId = jobId;
  match.isJob = { $ne: true };
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
        processingTime: d.processingTime || null,
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
const addCommonPersonIds = (rows = []) => {
  let counter = 0;
  const signatureMap = new Map();

  return rows.map((row) => {
    const object = String(row.object || "").toLowerCase();
    const isPerson = object.includes("person");
    if (!isPerson) return row;

    const signature = [
      String(row.ocrText || "").toLowerCase(),
      String(row.color || "").toLowerCase(),
      object,
    ].join("|");

    let commonPersonId = signatureMap.get(signature);
    if (!commonPersonId) {
      counter += 1;
      commonPersonId = `common_${counter}`;
      signatureMap.set(signature, commonPersonId);
    }

    return { ...row, commonPersonId };
  });
};

const processVideoInBackground = async (jobId, videoUrl, userId, userPrompt, imageUrl = null) => {
  try {
    const BASE_URL = process.env.BASE_URL;
    const startTime = Date.now();

    const response = await axios.post(
      `${process.env.PYTHON_API_URL}/process`,
      { fileUrl: videoUrl, imageUrl, prompt: userPrompt },
      { timeout: 7200000 }
    );

    const endTime = Date.now();
    const processingTimeStr = `${((endTime - startTime) / 1000).toFixed(2)}s`;

    const detections = (response.data.results || []).map((d, index) => {
      const cleanPath = (d.image_path || "").replace(/\\/g, "/");
      const fileNameOnly = cleanPath.split("/").pop();

      return {
        userId: new mongoose.Types.ObjectId(userId),
        jobId,
        status: "processing",
        fileName: videoUrl.split("/").pop(),
        textNote: userPrompt,
        object: d.object || d.prompt || "unknown",
        ocrText: d.ocr_text || "",
        confidence: d.confidence || d.similarity || 0.5,
        trackingId: `track_${Date.now()}_${index}`,
        timestamp: d.timestamp || 0,
        bbox: Array.isArray(d.bbox) ? d.bbox : [],
        imagePath: fileNameOnly,
        screenshotUrl: fileNameOnly
          ? `${BASE_URL}/uploads/detected_frames/${fileNameOnly}`
          : "",
        processingTime: processingTimeStr,
        videoUrl,
        color: d.color || "",
      };
    });

    if (detections.length > 0) {
      await Detection.insertMany(detections, { ordered: false });
    }

    // ✅ mark completed
    await Detection.updateMany(
      { jobId },
      { $set: { status: "completed" } }
    );

  } catch (err) {
    console.error("Background Error:", err.message);

    await Detection.updateMany(
      { jobId },
      { $set: { status: "failed" } }
    );
  }
};

exports.uploadAndProcess = async (req, res) => {
  try {
    const userId = req.userId;
    const videoFiles = req.files?.file || [];
    const userPrompt = req.body.text || "person";
    console.log("[uploadAndProcess] request", {
      userId,
      totalVideos: videoFiles.length,
      hasImage: Boolean(req.files?.image?.[0]),
      prompt: userPrompt,
    });

    if (!videoFiles.length) {
      return res.status(400).json({ message: "Video file missing" });
    }

    const BASE_URL = process.env.BASE_URL;
    const imageFile = req.files?.image?.[0];
    const imageUrl = imageFile ? `${BASE_URL}/uploads/${imageFile.filename}` : null;

    // Keep async job flow for single video (frontend already supports polling)
    if (videoFiles.length === 1) {
      const videoFile = videoFiles[0];
      const jobId = `job_${Date.now()}`;
      const videoUrl = `${BASE_URL}/uploads/${videoFile.filename}`;

      await Detection.create({
        userId: new mongoose.Types.ObjectId(userId),
        jobId,
        status: "processing",
        fileName: videoFile.filename,
        videoUrl,
        textNote: userPrompt,
        isJob: true,
      });

      processVideoInBackground(jobId, videoUrl, userId, userPrompt, imageUrl);
      console.log("[uploadAndProcess] single job created", { jobId, videoUrl });

      return res.json({
        message: "Upload successful",
        jobId,
        status: "processing",
      });
    }

    // Multi-video async: upload first, then process each video in background.
    const batchJobId = `job_${Date.now()}`;
    await Detection.create({
      userId: new mongoose.Types.ObjectId(userId),
      jobId: batchJobId,
      status: "processing",
      fileName: "__batch__",
      textNote: userPrompt,
      isJob: true,
    });

    videoFiles.forEach(async (videoFile, index) => {
      try {
        const childJobId = `${batchJobId}__v${index + 1}`;
        const videoUrl = `${BASE_URL}/uploads/${videoFile.filename}`;

        await Detection.create({
          userId: new mongoose.Types.ObjectId(userId),
          jobId: childJobId,
          status: "processing",
          fileName: videoFile.filename,
          videoUrl,
          textNote: userPrompt,
          isJob: true,
        });

        processVideoInBackground(childJobId, videoUrl, userId, userPrompt, imageUrl);
        console.log("[uploadAndProcess] child job queued", { batchJobId, childJobId, videoUrl });
      } catch (childErr) {
        console.error("Child job create error:", childErr.message);
      }
    });

    return res.json({
      message: "Upload successful",
      mode: "Multi Video Upload (Background)",
      jobId: batchJobId,
      status: "processing",
      totalVideos: videoFiles.length,
    });

  } catch (err) {
    console.error("Upload Error:", err);
    return res.status(500).json({ error: err.message });
  }
};
// ✅ IMAGE-ONLY DETECTION (no video)
exports.detectFromImage = async (req, res) => {
  try {
    const userId = req.userId;
    // Accept either `image` field or `file` field when it's actually an image.
    let imageFile = req.files?.image?.[0];
    if (!imageFile) {
      const maybe = req.files?.file?.[0];
      if (maybe && typeof maybe.mimetype === "string" && maybe.mimetype.startsWith("image/")) {
        imageFile = maybe;
      }
    }
    // If user doesn't provide prompt for image-only, run AUTO detect.
    const userPrompt = (req.body.text && String(req.body.text).trim()) ? req.body.text : "auto";

    if (!imageFile) {
      return res.status(400).json({ message: "Image file missing" });
    }

    const startTime = Date.now();
    const BASE_URL = process.env.BASE_URL;
    const imageUrl = `${BASE_URL}/uploads/${imageFile.filename}`;

    let response;
    try {
      // fileUrl intentionally empty => Python will run reference_image_only path
      response = await axios.post(
        `${process.env.PYTHON_API_URL}/process`,
        { fileUrl: "", imageUrl, prompt: userPrompt },
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

    const detections = (response.data.results || []).map((d, index) => {
      const cleanPath = (d.image_path || "").replace(/\\/g, "/");
      const fileNameOnly = cleanPath.split("/").pop();

      let conf = 0.50;
      if (typeof d.confidence === "number") conf = d.confidence;
      else if (typeof d.similarity === "number") conf = d.similarity;
      else if (typeof d.confidence === "string")
        conf = parseFloat(d.confidence) || 0.50;
      else if (typeof d.similarity === "string")
        conf = parseFloat(d.similarity) || 0.50;

      return {
        object: d.object || d.prompt || "unknown",
        ocrText: d.ocr_text || d.plate || d.ocrText || "",
        confidence: conf,
        processingTime: processingTimeStr,
        trackingId: (d.trackingId || `img_${Date.now()}_${index}`).toString(),
        timestamp: typeof d.timestamp === "number" ? d.timestamp : 0,
        bbox: Array.isArray(d.bbox) ? d.bbox : [],
        imagePath: fileNameOnly,
        color: d.color || "",
      };
    });

    const normalizedPrompt = userPrompt
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, "")
      .trim();

    return res.json({
      message: "Success",
      processing_time: processingTimeStr,
      mode: response.data.mode || "image_only",
      total_found: detections.length,
      totalUniqueObjects: new Set(detections.map((d) => d.trackingId)).size,
      counts: {
        [`total_${normalizedPrompt.replace(/\s+/g, "_")}`]: detections.length,
      },
      results: detections.map((d) => ({
        object: d.object,
        ocrText: d.ocrText,
        confidence: d.confidence,
        trackingId: d.trackingId,
        timestamp: d.timestamp,
        bbox: d.bbox,
        image_path: d.imagePath,
        screenshotUrl: d.imagePath
          ? `${BASE_URL}/uploads/detected_frames/${d.imagePath}`
          : "",
        color: d.color,
      })),
      source: "image_only",
      uploadedImageUrl: imageUrl,
    });
  } catch (err) {
    console.error("detectFromImage Error:", err);
    return res.status(500).json({ error: err.message });
  }
};

exports.deleteByTrackingId = async (req, res) => {
  try {
    const userId = req.userId;
    const { trackingId } = req.params;
    const { jobId } = req.query; // ✅ optional

    console.log("User:", userId, "Tracking:", trackingId, "Job:", jobId);

    if (!trackingId) {
      return res.status(400).json({ message: "trackingId is required" });
    }

    let filter = {
      userId: new mongoose.Types.ObjectId(userId),
      trackingId,
    };

    // 🔥 OPTIONAL: same trackingId different jobs me ho to control
    if (jobId) {
      filter.jobId = jobId;
    }

    const result = await Detection.deleteMany(filter);

    return res.json({
      message: "Detections deleted successfully",
      deletedCount: result.deletedCount,
    });

  } catch (err) {
    console.error("deleteByTrackingId Error:", err);
    return res.status(500).json({ error: err.message });
  }
};

exports.deleteMultiple = async (req, res) => {
  try {
    const userId = req.userId;
    const { jobId } = req.query;
    const trackingIds = req.body?.trackingIds || [];

    if (!userId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    let filter = {
      userId: new mongoose.Types.ObjectId(userId),
    };

    // 🔥 Case 1: delete by jobId
    if (jobId) {
      filter.jobId = jobId;
    }

    // 🔥 Case 2: delete selected trackingIds
    else if (trackingIds.length > 0) {
      filter.trackingId = { $in: trackingIds };
    }

    // 🔥 Case 3: DELETE ALL (no extra condition needed)
    // filter remains only userId

    const result = await Detection.deleteMany(filter);

    return res.json({
      message: "Deleted successfully",
      deletedCount: result.deletedCount,
    });

  } catch (err) {
    console.error("Delete Error:", err);
    return res.status(500).json({ error: err.message });
  }
};

exports.getJobStatus = async (req, res) => {
  try {
    const { jobId } = req.params;
    const userId = req.userId;
    console.log("[getJobStatus] checking", { userId, jobId });

    const job = await Detection.findOne({
      jobId,
      userId: new mongoose.Types.ObjectId(userId),
        isJob: true, // 🔥 ADD THIS

    });

    if (!job) {
      return res.json({ jobId, status: "not_found", results: [] });
    }

    let status = job.status;
    let results = [];

    // Batch parent job: aggregate all child jobs.
    if (job.fileName === "__batch__") {
      const childJobs = await Detection.find({
        userId: new mongoose.Types.ObjectId(userId),
        isJob: true,
        jobId: { $regex: `^${jobId}__v` },
      }).sort({ createdAt: 1 });

      const allCompleted = childJobs.length > 0 && childJobs.every((j) => j.status === "completed");
      const anyFailed = childJobs.some((j) => j.status === "failed");

      if (allCompleted) {
        status = "completed";
        const childIds = childJobs.map((j) => j.jobId);
        results = await Detection.find({
          userId: new mongoose.Types.ObjectId(userId),
          isJob: { $ne: true },
          jobId: { $in: childIds },
        }).sort({ createdAt: -1 });
        await Detection.updateOne({ _id: job._id }, { $set: { status: "completed" } });
      } else if (anyFailed) {
        status = "failed";
      } else {
        status = "processing";
      }
    } else if (status === "completed") {
      results = await Detection.find({
        jobId,
        userId: new mongoose.Types.ObjectId(userId),
        isJob: { $ne: true },
      }).sort({ createdAt: -1 });
    }

    const resultRows = addCommonPersonIds(
      results.map((d) => ({
        object: d.object,
        ocrText: d.ocrText,
        confidence: d.confidence,
        trackingId: d.trackingId,
        timestamp: d.timestamp,
        image_path: d.imagePath,
        bbox: d.bbox,
        screenshotUrl: d.screenshotUrl,
        color: d.color || "",
      })),
    );

    return res.json({
      jobId,
      status,
      total: resultRows.length,
      totalCommonPersons: new Set(resultRows.map((r) => r.commonPersonId).filter(Boolean)).size,
      results: resultRows,
    });

  } catch (err) {
    console.error("Status Error:", err);
    return res.status(500).json({ error: err.message });
  }
};

exports.getHistory = async (req, res) => {
  try {
    const userId = req.userId;
    const history = await Detection.aggregate([
      {
        $match: {
          userId: new mongoose.Types.ObjectId(userId),
          isJob: { $ne: true },
        },
      },
      { $sort: { createdAt: -1 } },
      {
        $group: {
          _id: {
            $ifNull: [
              "$jobId",
              { $concat: ["file_", { $ifNull: ["$fileName", "unknown"] }] },
            ],
          },
          createdAt: { $max: "$createdAt" },
          videoUrl: { $first: "$videoUrl" },
          prompt: { $first: "$textNote" },
          processingTime: { $first: "$processingTime" },
          status: { $first: "$status" },
          detections: {
            $push: {
              _id: "$_id",
              object: "$object",
              confidence: "$confidence",
              trackingId: "$trackingId",
              timestamp: "$timestamp",
              image: "$screenshotUrl",
              bbox: "$bbox",
              color: "$color",
            },
          },
        },
      },
      { $sort: { createdAt: -1 } },
    ]);

    const formatted = history.map((item) => ({
      _id: String(item._id),
      videoUrl: item.videoUrl || "",
      prompt: item.prompt || "",
      processingTime: item.processingTime || "N/A",
      status: item.status || "completed",
      totalDetections: Array.isArray(item.detections) ? item.detections.length : 0,
      detections: item.detections || [],
      createdAt: item.createdAt,
    }));

    return res.json({ history: formatted });
  } catch (err) {
    console.error("getHistory Error:", err);
    return res.status(500).json({ error: err.message });
  }
};

exports.deleteHistoryEntry = async (req, res) => {
  try {
    const userId = req.userId;
    const { entryId } = req.params;

    const result = await Detection.deleteMany({
      userId: new mongoose.Types.ObjectId(userId),
      isJob: { $ne: true },
      $or: [{ jobId: entryId }, { _id: entryId }],
    });

    return res.json({ message: "History entry deleted", deletedCount: result.deletedCount });
  } catch (err) {
    console.error("deleteHistoryEntry Error:", err);
    return res.status(500).json({ error: err.message });
  }
};

exports.deleteHistoryBatch = async (req, res) => {
  try {
    const userId = req.userId;
    const ids = Array.isArray(req.body?.ids) ? req.body.ids : [];

    if (!ids.length) {
      return res.status(400).json({ message: "ids are required" });
    }

    const result = await Detection.deleteMany({
      userId: new mongoose.Types.ObjectId(userId),
      isJob: { $ne: true },
      $or: [{ jobId: { $in: ids } }, { _id: { $in: ids } }],
    });

    return res.json({ message: "History entries deleted", deletedCount: result.deletedCount });
  } catch (err) {
    console.error("deleteHistoryBatch Error:", err);
    return res.status(500).json({ error: err.message });
  }
};

exports.clearHistory = async (req, res) => {
  try {
    const userId = req.userId;
    const result = await Detection.deleteMany({
      userId: new mongoose.Types.ObjectId(userId),
      isJob: { $ne: true },
    });

    return res.json({ message: "History cleared", deletedCount: result.deletedCount });
  } catch (err) {
    console.error("clearHistory Error:", err);
    return res.status(500).json({ error: err.message });
  }
};