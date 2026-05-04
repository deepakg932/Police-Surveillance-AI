const axios = require("axios");
const Detection = require("../models/Detection.js");
const mongoose = require("mongoose")
const path = require("path");

const activePythonJobs = new Set();
global.activePythonJobs = activePythonJobs;
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
exports.uploadAndProcess = async (req, res) => {
  try {
    const userId = req.userId;
    const videoFile = req.files?.file?.[0];
    const imageFile = req.files?.image?.[0];
    const userPrompt = req.body.text || "person";

    if (!videoFile) {
      return res.status(400).json({ message: "Video file missing" });
    }

    const BASE_URL = process.env.BASE_URL;
    const videoUrl = `${BASE_URL}/uploads/${videoFile.filename}`;
    const imageUrl = imageFile ? `${BASE_URL}/uploads/${imageFile.filename}` : null;

    // ─── STEP 1: Start Python job first, get real jobId ───────
    let jobId;
    try {
      const startRes = await axios.post(
        `${process.env.PYTHON_API_URL}/process/start`,
        { fileUrl: videoUrl, imageUrl, prompt: userPrompt },
        { timeout: 30000 }
      );
      jobId = startRes.data.job_id;
      console.log("✅ Python job started:", jobId);
    } catch (err) {
      console.error("❌ Failed to start Python job:", err.message);
      return res.status(500).json({ error: "Python job start failed", details: err.message });
    }

    // ─── STEP 2: Save initial job entry ───────────────────────
    await Detection.create({
      userId: new mongoose.Types.ObjectId(userId),
      jobId,
      isJob: true,
      status: "processing",
      fileName: videoFile.filename,
      textNote: userPrompt,
      videoUrl,
      processingTime: null,
    });

    // ─── STEP 3: Background polling + save ────────────────────
    processVideoInBackground(jobId, videoUrl, userId, userPrompt, videoFile.filename);

    // ─── STEP 4: Immediate response (no timeout!) ─────────────
    return res.json({
      message: "Upload successful",
      jobId,
      status: "processing",
    });

  } catch (err) {
    console.error("Upload Error:", err);
    return res.status(500).json({ error: err.message });
  }
};


const processVideoInBackground = async (
  jobId,
  videoUrl,
  userId,
  userPrompt,
  fileName,
) => {
  try {
    activePythonJobs.add(jobId);
    const BASE_URL = process.env.BASE_URL;
    const startTime = Date.now();

    const MAX_WAIT_MS = 7200000;
    const POLL_INTERVAL = 10000;
    const pollStart = Date.now();

    let pyResult = null;
    let retryCount = 0;
    const MAX_RETRY = 3;

    while (Date.now() - pollStart < MAX_WAIT_MS) {
      await new Promise((r) => setTimeout(r, POLL_INTERVAL));

      let statusRes;
      try {
        statusRes = await axios.get(
          `${process.env.PYTHON_API_URL}/process/status/${jobId}`,
          { timeout: 15000 }
        );
      } catch (pollErr) {
        retryCount++;

        console.warn(`❌ Poll failed (${retryCount})`, pollErr.message);

        if (retryCount >= MAX_RETRY) {
          throw new Error("Python server unreachable / network failed");
        }

        continue;
      }

      const { status, result, error } = statusRes.data;
      retryCount = 0;
      console.log(`🔄 Job ${jobId} status: ${status}`);

      const liveResults = result?.results || statusRes.data?.results || [];

      if (Array.isArray(liveResults) && liveResults.length > 0) {
        const liveDocs = liveResults.map((d, index) => {
          const cleanPath = (d.image_path || d.imagePath || "").replace(/\\/g, "/");
          const fileNameOnly = cleanPath.split("/").pop();

          let conf = 0.85;
          if (typeof d.confidence === "number") conf = d.confidence;
          else if (d.plate) conf = 0.9;

          let bbox = [];
          for (const key of ["bbox", "plate_bbox", "person_bbox", "vehicle_bbox"]) {
            if (Array.isArray(d[key]) && d[key].length === 4) {
              bbox = d[key].map(Number).filter((n) => !isNaN(n));
              if (bbox.length !== 4) bbox = [];
              break;
            }
          }

          return {
            userId: new mongoose.Types.ObjectId(userId),
            jobId,
            isJob: false,
            status: "processing",
            fileName,
            textNote: userPrompt,
            object: d.object || d.prompt || "unknown",
            ocrText: d.ocr_text || d.plate || d.ocrText || "",
            confidence: conf,
            trackingId: (d.trackingId || d.tracking_id || `live_${jobId}_${index}`).toString(),
            timestamp: typeof d.timestamp === "number" ? d.timestamp : 0,
            bbox,
            imagePath: fileNameOnly,
            screenshotUrl: d.screenshotUrl || (fileNameOnly ? `${BASE_URL}/${fileNameOnly}` : ""),
            processingTime: "",
            videoUrl,
            color: d.color || d.vehicle || "",
          };
        });

        await Detection.bulkWrite(
          liveDocs.map((doc) => ({
            updateOne: {
              filter: {
                userId: doc.userId,
                jobId: doc.jobId,
                trackingId: doc.trackingId,
                isJob: false,
              },
              update: { $set: doc },
              upsert: true,
            },
          }))
        );

        console.log(`🟢 Live detections saved: ${liveDocs.length}`);
      }

      if (status === "completed") {
        pyResult = result;
        break;
      }

      if (status === "failed") {
        throw new Error(`Python job failed: ${error || "unknown"}`);
      }
    }

    if (!pyResult) {
      throw new Error("Timeout after 2 hours");
    }

    const endTime = Date.now();
    const processingTimeStr = `${((endTime - startTime) / 1000).toFixed(2)}s`;

    const detections = (pyResult.results || []).map((d, index) => {
      const cleanPath = (d.image_path || d.imagePath || "").replace(/\\/g, "/");
      const fileNameOnly = cleanPath.split("/").pop();

      let conf = 0.85;
      if (typeof d.confidence === "number") conf = d.confidence;
      else if (d.plate) conf = 0.9;

      let bbox = [];
      for (const key of ["bbox", "plate_bbox", "person_bbox", "vehicle_bbox"]) {
        if (Array.isArray(d[key]) && d[key].length === 4) {
          bbox = d[key].map(Number).filter((n) => !isNaN(n));
          if (bbox.length !== 4) bbox = [];
          break;
        }
      }

      return {
        userId: new mongoose.Types.ObjectId(userId),
        jobId,
        isJob: false,
        status: "completed",
        fileName,
        textNote: userPrompt,
        object: d.object || d.prompt || "unknown",
        ocrText: d.ocr_text || d.plate || d.ocrText || "",
        confidence: conf,
        trackingId: (d.trackingId || d.tracking_id || `TRK-${Math.floor(1000 + Math.random() * 9000)}`).toString(),
        timestamp: typeof d.timestamp === "number" ? d.timestamp : 0,
        bbox,
        imagePath: fileNameOnly,
        screenshotUrl: d.screenshotUrl || (fileNameOnly ? `${BASE_URL}/${fileNameOnly}` : ""),
        processingTime: processingTimeStr,
        videoUrl,
        color: d.color || d.vehicle || "",
      };
    });

    if (detections.length > 0) {
      await Detection.bulkWrite(
        detections.map((doc) => ({
          updateOne: {
            filter: {
              userId: doc.userId,
              jobId: doc.jobId,
              trackingId: doc.trackingId,
              isJob: false,
            },
            update: { $set: doc },
            upsert: true,
          },
        }))
      );

      console.log("💾 Final detections saved:", detections.length);
    }

    await Detection.deleteMany({
      jobId,
      userId: new mongoose.Types.ObjectId(userId),
      isJob: false,
      status: "processing",
    });

    await Detection.updateOne(
      { jobId, isJob: true },
      {
        $set: {
          status: "completed",
          processingTime: processingTimeStr,
          mode: pyResult.mode || null,
        },
      }
    );

    console.log(`✅ Background job done: ${jobId} — ${detections.length} detections`);
  } catch (err) {
    console.error("Background Error:", err.message);


    await Detection.updateOne(
      { jobId, isJob: true },
      {
        $set: {
          status: "failed",
          errorMessage: err.message,
        },
      }
    );
  } finally {
    activePythonJobs.delete(jobId);
    console.log("🧹 Job removed from active list:", jobId);
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
    const { jobId } = req.query;

    let filter = {
      userId: new mongoose.Types.ObjectId(userId),
    };

    // ✅ jobId hai to usse delete karo (job + uski detections dono)
    if (jobId) {
      filter.jobId = jobId;
      filter.trackingId = trackingId;
    } else {
      filter.trackingId = trackingId;
    }

    const result = await Detection.deleteMany(filter);
    return res.json({ message: "Deleted", deletedCount: result.deletedCount });

  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
};

exports.deleteMultiple = async (req, res) => {
  try {
    const userId = req.userId;
    const { jobId } = req.query;
    const jobIds = req.body?.jobIds || [];      // ✅ jobIds lo, trackingIds nahi
    const trackingIds = req.body?.trackingIds || [];

    let filter = { userId: new mongoose.Types.ObjectId(userId) };

    if (jobId) {
      // Case 1: ek specific job delete
      filter.jobId = jobId;
    } else if (jobIds.length > 0) {
      // Case 2: multiple jobs delete — job entries + detections dono
      filter.jobId = { $in: jobIds };
    } else if (trackingIds.length > 0) {
      // Case 3: sirf specific detections delete
      filter.trackingId = { $in: trackingIds };
    }
    // Case 4: kuch nahi — DELETE ALL (filter = sirf userId)

    const result = await Detection.deleteMany(filter);
    return res.json({ message: "Deleted", deletedCount: result.deletedCount });

  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
};


exports.getJobStatus = async (req, res) => {
  try {
    const { jobId } = req.params;
    const userId = req.userId;

    const userObjectId = new mongoose.Types.ObjectId(userId);

    const job = await Detection.findOne({
      jobId,
      userId: userObjectId,
      isJob: true,
    });

    if (!job) {
      return res.json({
        jobId,
        status: "not_found",
        total: 0,
        results: [],
      });
    }

    // ✅ processing me bhi detections return karega
    // const results = await Detection.find({
    //   jobId,
    //   userId: userObjectId,
    //   isJob: { $ne: true },
    // }).sort({ createdAt: -1 });
    const results = await Detection.find({
      jobId,
      userId: userObjectId,
      isJob: { $ne: true },
      ...(job.status === "completed" ? { status: "completed" } : {}),
    }).sort({ confidence: -1, createdAt: -1 });

    return res.json({
      message: "Success",
      jobId,
      processing_time: job.processingTime || null,
      status: job.status,
      errorMessage: job.errorMessage || "",
      total: results.length,
      results: results.map((d) => ({
        object: d.object,
        ocrText: d.ocrText,
        confidence: d.confidence,
        trackingId: d.trackingId,
        timestamp: d.timestamp,
        image_path: d.imagePath,
        bbox: d.bbox,
        screenshotUrl: d.screenshotUrl,
        color: d.color || "",
        action: d.action || "Clip Saved",
        status: d.status || job.status,
      })),
    });
  } catch (err) {
    console.error("Status Error:", err);
    return res.status(500).json({ error: err.message });
  }
};

// ================================================================
// MULTI VIDEO UPLOAD — Single video jaisa hi kaam karta hai
// Har video ko same prompt/image se process karta hai
// ================================================================
    const processBatchVideos = async (jobs, batchId, userId, BASE_URL) => {
  console.log(`🎬 Batch ${batchId} start — ${jobs.length} videos`);

  for (let i = 0; i < jobs.length; i++) {
    const job = jobs[i];
    let pyJobId = null;

    try {
      const startTime = Date.now();

      const pythonPayload = {
        fileUrl: job.videoUrl,
        prompt: job.userPrompt,
      };

      if (job.imageUrl) pythonPayload.imageUrl = job.imageUrl;

      const startRes = await axios.post(
        `${process.env.PYTHON_API_URL}/process/start`,
        pythonPayload,
        { timeout: 30000 }
      );

      pyJobId = startRes.data.job_id;
      activePythonJobs.add(pyJobId);
      console.log(`✅ Python job started: ${pyJobId}`);

      const POLL_INTERVAL_MS = 10000;
      const MAX_WAIT_MS = 7200000;
      const pollStart = Date.now();

      let pythonResult = null;
      let retryCount = 0;
      const MAX_RETRY = 3;

      while (Date.now() - pollStart < MAX_WAIT_MS) {
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));

        let statusRes;
        try {
          statusRes = await axios.get(
            `${process.env.PYTHON_API_URL}/process/status/${pyJobId}`,
            { timeout: 15000 }
          );
          retryCount = 0;
        } catch (pollErr) {
          retryCount++;
          console.warn(`Poll failed ${retryCount}/${MAX_RETRY}:`, pollErr.message);

          if (retryCount >= MAX_RETRY) {
            throw new Error("Python server unreachable / network failed");
          }
          continue;
        }

        const { status, result, error } = statusRes.data;
        const liveResults = result?.results || statusRes.data?.results || [];

        if (Array.isArray(liveResults) && liveResults.length > 0) {
          const liveDocs = liveResults.map((d, index) => {
            const cleanPath = (d.image_path || d.imagePath || "").replace(/\\/g, "/");
            const fileNameOnly = cleanPath.split("/").pop();

            return {
              userId: new mongoose.Types.ObjectId(userId),
              jobId: job.jobId,
              pyJobId,
              batchId,
              isJob: false,
              status: "processing",
              fileName: job.fileName,
              textNote: job.userPrompt,
              object: d.object || d.prompt || "unknown",
              ocrText: d.ocr_text || d.plate || d.ocrText || "",
              confidence: typeof d.confidence === "number" ? d.confidence : 0.85,
              trackingId: (d.trackingId || d.tracking_id || `live_${job.jobId}_${index}`).toString(),
              timestamp: typeof d.timestamp === "number" ? d.timestamp : 0,
              bbox: Array.isArray(d.bbox) && d.bbox.length === 4 ? d.bbox.map(Number) : [],
              imagePath: fileNameOnly,
              screenshotUrl: d.screenshotUrl || (fileNameOnly ? `${BASE_URL}/${fileNameOnly}` : ""),
              processingTime: "",
              videoUrl: job.videoUrl,
              color: d.color || d.vehicle || "",
            };
          });

          await Detection.bulkWrite(
            liveDocs.map((doc) => ({
              updateOne: {
                filter: {
                  userId: doc.userId,
                  batchId: doc.batchId,
                  jobId: doc.jobId,
                  trackingId: doc.trackingId,
                  isJob: false,
                },
                update: { $set: doc },
                upsert: true,
              },
            }))
          );
        }

        if (status === "completed") {
          pythonResult = result;
          break;
        }

        if (status === "failed") {
          throw new Error(`Python job failed: ${error || "unknown"}`);
        }
      }

      if (!pythonResult) throw new Error("Python job timed out after 2 hours");

      const processingTimeStr = `${((Date.now() - startTime) / 1000).toFixed(2)}s`;

      const detections = (pythonResult.results || []).map((d, index) => {
        const cleanPath = (d.image_path || d.imagePath || "").replace(/\\/g, "/");
        const fileNameOnly = cleanPath.split("/").pop();

        return {
          userId: new mongoose.Types.ObjectId(userId),
          jobId: job.jobId,
          batchId,
          isJob: false,
          status: "completed",
          fileName: job.fileName,
          textNote: job.userPrompt,
          object: d.object || d.prompt || "unknown",
          ocrText: d.ocr_text || d.plate || d.ocrText || "",
          confidence: typeof d.confidence === "number" ? d.confidence : 0.85,
          trackingId: (d.trackingId || d.tracking_id || `TRK-${Math.floor(1000 + Math.random() * 9000)}`).toString(),
          timestamp: typeof d.timestamp === "number" ? d.timestamp : 0,
          bbox: Array.isArray(d.bbox) && d.bbox.length === 4 ? d.bbox.map(Number) : [],
          imagePath: fileNameOnly,
          screenshotUrl: fileNameOnly ? `${BASE_URL}/${fileNameOnly}` : "",
          processingTime: processingTimeStr,
          videoUrl: job.videoUrl,
          color: d.color || d.vehicle || "",
        };
      });

      if (detections.length > 0) {
        await Detection.bulkWrite(
          detections.map((doc) => ({
            updateOne: {
              filter: {
                userId: doc.userId,
                batchId: doc.batchId,
                jobId: doc.jobId,
                trackingId: doc.trackingId,
                isJob: false,
              },
              update: { $set: doc },
              upsert: true,
            },
          }))
        );
      }

      await Detection.deleteMany({
        batchId,
        jobId: job.jobId,
        userId: new mongoose.Types.ObjectId(userId),
        isJob: false,
        status: "processing",
      });

      await Detection.updateOne(
        { jobId: job.jobId, isJob: true },
        { $set: { status: "completed", processingTime: processingTimeStr } }
      );

    } catch (err) {
      await Detection.updateOne(
        { jobId: job.jobId, isJob: true },
        { $set: { status: "failed", errorMessage: err.message } }
      );
    } finally {
      if (pyJobId) {
        activePythonJobs.delete(pyJobId);
        console.log("🧹 Removed multi python job:", pyJobId);
      }
    }
  }

  const completedJobs = await Detection.find({
    batchId,
    isJob: true,
    isBatch: { $ne: true },
    status: "completed",
  });

  const failedJobs = await Detection.countDocuments({
    batchId,
    isJob: true,
    isBatch: { $ne: true },
    status: "failed",
  });

  const totalSeconds = completedJobs.reduce((sum, j) => {
    return sum + (parseFloat(j.processingTime) || 0);
  }, 0);

 await Detection.updateOne(
  { jobId: batchId, isBatch: true },
  {
    $set: {
      status: failedJobs > 0 && completedJobs.length === 0 ? "failed" : "completed",
      processingTime: `${totalSeconds.toFixed(2)}s`,
      errorMessage:
        failedJobs > 0 && completedJobs.length === 0
          ? "Python server unreachable / network failed"
          : "",
    },
  }
);
};




















exports.uploadAndProcessMultiple = async (req, res) => {
  try {
    const userId = req.userId;
    const videoFiles = req.files?.files || [];           // "files" field
    const imageFile = req.files?.image?.[0] || null;    // optional image (same as single)
    const userPrompt = req.body.text || "person";

    if (!videoFiles.length) {
      return res.status(400).json({ message: "Koi video file nahi mili" });
    }

    const BASE_URL = process.env.BASE_URL;
    const batchId = `batch_${Date.now()}`;

    // Image URL — same as single video logic
    const imageUrl = imageFile
      ? `${BASE_URL}/uploads/${imageFile.filename}`
      : null;

    const jobEntries = [];
    const jobs = [];

    for (const videoFile of videoFiles) {
      const jobId = `job_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
      const videoUrl = `${BASE_URL}/uploads/${videoFile.filename}`;

      // Exactly single video jaisa job entry
      jobEntries.push({
        userId: new mongoose.Types.ObjectId(userId),
        jobId,
        batchId,
        status: "processing",
        fileName: videoFile.filename,
        videoUrl,
        textNote: userPrompt,
        isJob: true,
      });

      jobs.push({ jobId, videoUrl, imageUrl, userPrompt, fileName: videoFile.filename });
    }

    // Batch tracker
    jobEntries.push({
      userId: new mongoose.Types.ObjectId(userId),
      jobId: batchId,
      batchId,
      status: "processing",
      fileName: `batch_${videoFiles.length}_videos`,
      textNote: userPrompt,
      isJob: true,
      isBatch: true,
      totalVideos: videoFiles.length,
    });

    await Detection.insertMany(jobEntries, { ordered: false });

    // Background me process karo — exactly single video jaisa
    processBatchVideos(jobs, batchId, userId, BASE_URL);

    return res.json({
      message: "Upload successful — processing shuru ho gaya",
      batchId,
      totalVideos: videoFiles.length,
      status: "processing",
      prompt: userPrompt,
      hasImage: !!imageFile,
    });

  } catch (err) {
    console.error("Multi-upload Error:", err);
    return res.status(500).json({ error: err.message });
  }
};

// Internal: har video ko EXACTLY single video jaisa process karo
// Internal: har video ko EXACTLY single video jaisa process karo


    exports.getHistory = async (req, res) => {
      try {
        const userId = req.userId;
        // ✅ AUTO FIX STUCK PROCESSING JOBS
        await Detection.updateMany(
          {
            userId: new mongoose.Types.ObjectId(userId),
            isJob: true,
            status: "processing",
            createdAt: { $lt: new Date(Date.now() - 5 * 60 * 1000) }, // 5 min old
          },
          {
            $set: {
              status: "failed",
              errorMessage: "Processing stopped (network / server issue)",
            },
          }
        );

        const jobs = await Detection.find({
          userId: new mongoose.Types.ObjectId(userId),
          isJob: true,
          isBatch: { $ne: true },
          status: "completed", // ✅ only completed history show
        }).sort({ createdAt: -1 });

        const history = await Promise.all(
          jobs.map(async (job) => {
            let results = [];

            if (job.status === "completed") {
              const detections = await Detection.find({
                jobId: job.jobId,
                userId: new mongoose.Types.ObjectId(userId),
                isJob: { $ne: true },
              }).sort({ createdAt: -1 });

              results = detections.map((d) => ({
                object: d.object,
                ocrText: d.ocrText || "",
                confidence: d.confidence,
                trackingId: d.trackingId,
                timestamp: d.timestamp,
                image_path: d.imagePath,
                bbox: d.bbox,
                screenshotUrl: d.screenshotUrl,
                color: d.color || "",
              }));
            }

            const totalUniqueObjects = new Set(results.map((r) => r.trackingId)).size;
            const counts = job.textNote
              ? {
                [`total_${job.textNote.toLowerCase().replace(/\s+/g, "_")}`]:
                  results.length,
              }
              : {};

            return {
              // ✅ Top-level fields matching processing response
              message: "Success",
              processing_time: job.processingTime || null,
              mode: job.mode || null,
              total_found: results.length,
              totalUniqueObjects,
              counts,

              // ✅ Job meta
              id: job.jobId,
              jobId: job.jobId,
              videoName: job.fileName,
              prompt: job.textNote,
              status: job.status,
              timestamp: job.createdAt,
              videoUrl: job.videoUrl,
              total: results.length,

              // ✅ Full results array (same shape as processing response)
              results,
            };
          })
        );

        return res.json({ history });
      } catch (err) {
        console.error("History Error:", err);
        return res.status(500).json({ error: err.message });
      }
    };

    exports.deleteHistoryEntry = async (req, res) => {
      try {
        const userId = req.userId;
        const { entryId } = req.params;

        // ✅ Hamesha jobId se delete karo — simple aur safe
        const result = await Detection.deleteMany({
          userId: new mongoose.Types.ObjectId(userId),
          jobId: entryId,
        });

        return res.json({ message: "Deleted", deletedCount: result.deletedCount });

      } catch (err) {
        console.error("deleteHistoryEntry Error:", err);
        return res.status(500).json({ error: err.message });
      }
    };

    // ✅ Multiple jobs delete
    exports.deleteHistoryBatch = async (req, res) => {
      try {
        const userId = req.userId;
        const ids = Array.isArray(req.body?.ids) ? req.body.ids : [];

        if (!ids.length) {
          return res.status(400).json({ message: "ids array is required" });
        }

        // ids = array of jobIds e.g. ["job_123", "job_456"]
        const result = await Detection.deleteMany({
          userId: new mongoose.Types.ObjectId(userId),
          jobId: { $in: ids },
        });

        return res.json({
          message: "History entries deleted",
          deletedCount: result.deletedCount,
        });
      } catch (err) {
        console.error("deleteHistoryBatch Error:", err);
        return res.status(500).json({ error: err.message });
      }
    };

    // ✅ Clear ALL history (job entries + detections sab)
    exports.clearHistory = async (req, res) => {
      try {
        const userId = req.userId;

        const result = await Detection.deleteMany({
          userId: new mongoose.Types.ObjectId(userId),
        });

        return res.json({
          message: "History cleared",
          deletedCount: result.deletedCount,
        });
      } catch (err) {
        console.error("clearHistory Error:", err);
        return res.status(500).json({ error: err.message });
      }
    };


    exports.downloadDetectionImage = async (req, res) => {
      try {
        const { file } = req.query;

        if (!file) {
          return res.status(400).json({ message: "file is required" });
        }

        const safeFile = path.basename(file);
        const filePath = path.join(
          "/var/www/workingcart__usr/data/www/workingcart.com/Police-Surveillance-AI/backend/ai-python/detected_frames",
          safeFile
        );

        return res.download(filePath, safeFile);
      } catch (err) {
        return res.status(500).json({ error: err.message });
      }
    };

    exports.markJobFailed = async (req, res) => {
      try {
        const { jobId } = req.params;

        await Detection.updateOne(
          { jobId, isJob: true },
          {
            $set: {
              status: "failed",
              errorMessage: "Status API failed / network error",
            },
          }
        );

        return res.json({ message: "Job marked failed", jobId });
      } catch (err) {
        return res.status(500).json({ error: err.message });
      }
    };






exports.getBatchStatus = async (req, res) => {
      `         `
      try {
        const { batchId } = req.params;
        const userId = req.userId;

        const batch = await Detection.findOne({
          jobId: batchId,
          isBatch: true,
          userId: new mongoose.Types.ObjectId(userId),
        });

        if (!batch) {
          return res.json({ batchId, status: "not_found" });
        }

        const individualJobs = await Detection.find({
          batchId,
          isJob: true,
          isBatch: { $ne: true },
          userId: new mongoose.Types.ObjectId(userId),
        });

        const jobSummary = individualJobs.map((j) => ({
          jobId: j.jobId,
          fileName: j.fileName,
          status: j.status,
          processing_time: j.processingTime || null,
        }));

        const completed = jobSummary.filter((j) => j.status === "completed").length;
        const failed = jobSummary.filter((j) => j.status === "failed").length;
        const total = jobSummary.length;

        let results = [];
        if (completed > 0 || batch.status === "processing") {
          const activeJobIds = jobSummary
            .filter((j) => j.status === "completed" || j.status === "processing")
            .map((j) => j.jobId);

          const detections = await Detection.find({
            batchId,
            isJob: { $ne: true },
            userId: new mongoose.Types.ObjectId(userId),
            jobId: { $in: activeJobIds },
            ...(batch.status === "completed" ? { status: "completed" } : {}),
          }).sort({ confidence: -1, createdAt: -1 });

          results = detections.map((d) => ({
            object: d.object,
            ocrText: d.ocrText,
            confidence: d.confidence,
            trackingId: d.trackingId,
            timestamp: d.timestamp,
            image_path: d.imagePath,
            bbox: d.bbox,
            screenshotUrl: d.screenshotUrl,
            fileName: d.fileName,
            jobId: d.jobId,
            color: d.color || "",
            processing_time: d.processingTime || null,
          }));
        }

        return res.json({
  message: "Success",
  jobId: batchId,
  batchId,
  processing_time: batch.processingTime || null,
  status: batch.status,
  errorMessage:
    batch.errorMessage ||
    (failed > 0 ? "Python server unreachable / network failed" : ""),
  total: results.length,
  results,
  totalVideos: total,
  completed,
  failed,
  pending: total - completed - failed,
  isAllDone: completed + failed === total && total > 0,
  jobs: jobSummary,
});
      } catch (err) {
        console.error("getBatchStatus Error:", err);
        return res.status(500).json({ error: err.message });
      }
    };