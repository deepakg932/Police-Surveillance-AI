const axios = require("axios");
const Detection = require("../models/Detection.js");
const path = require("path");

exports.searchDetections = async (req, res) => {
  try {
    const { object, textNote, trackingId, fileName } = req.query;
    let filter = {};

    // 1. Filter Setup (Regex for dynamic search)
    if (object) filter.object = { $regex: object, $options: "i" };
    if (textNote) filter.textNote = { $regex: textNote, $options: "i" };
    if (trackingId) filter.trackingId = trackingId;
    if (fileName) filter.fileName = fileName;

    // 2. Database se results nikaalo
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
        image_path: d.imagePath, // DB field name match karo
        bbox: d.bbox,
        processing_time: d.processingTime,
        screenshotUrl: d.screenshotUrl, // DB se direct URL
      })),
    });
  } catch (err) {
    console.error("❌ Search Error:", err.message);
    return res.status(500).json({ error: err.message });
  }
};

// exports.uploadAndProcess = async (req, res) => {
//   try {
//     const videoFile = req.files?.file?.[0];
//     const imageFile = req.files?.image?.[0]; 
//     const userPrompt = req.body.text || "person, car";

//     if (!videoFile) return res.status(400).json({ message: "Video missing" });
    
//     const startTime = Date.now();

//     // URLs for Python to download the files
//     const videoUrl = `${process.env.BASE_URL}/uploads/${videoFile.filename}`;
//     const imageUrl = imageFile 
//       ? `${process.env.BASE_URL}/uploads/${imageFile.filename}` 
//       : null;

//     // Call Python API
//     const response = await axios.post(
//       `${process.env.PYTHON_API_URL}/process`,
//       {
//         fileUrl: videoUrl,
//         imageUrl: imageUrl,
//         prompt: userPrompt,
//       },
//       {
//         timeout: 0, // Important for long videos
//         maxContentLength: Infinity,
//         maxBodyLength: Infinity,
//       }
//     );

//     const endTime = Date.now();
//     const processingTimeStr = `${((endTime - startTime) / 1000).toFixed(2)}s`;

//     // 🎯 UPDATE: Sync threshold with Python (Python uses 0.55, Node filters at 0.55)
//     // const detections = (response.data.results || []).filter(
//     //   (d) => d.confidence >= 0.55 
//     // );


//     const detections = (response.data.results || []).filter((d) => d.confidence >= 0.35);
//     const BASE_URL = process.env.BASE_URL;
//     const finalCounts = {};

//     // Generate dynamic counts based on prompt keywords
//     const keywords = userPrompt
//       .toLowerCase()
//       .split(/[\s,]+/)
//       .filter((w) => !["with", "and", "wearing", "in", "a", "wear"].includes(w));

//     keywords.forEach((key) => {
//       const uniqueSet = new Set();
//       detections.forEach((d) => {
//         if (d.object.toLowerCase().includes(key)) {
//           uniqueSet.add(d.trackingId);
//         }
//       });
//       finalCounts[`total_${key}`] = uniqueSet.size;
//     });

//     // Save to MongoDB
//     if (detections.length > 0) {
//       const detectionsToSave = detections.map((d) => {
//         const cleanPath = d.image_path.replace(/\\/g, "/");
//         return {
//           fileName: videoFile.filename,
//           textNote: userPrompt,
//           object: d.object,
//           ocrText: d.ocr_text, // 👈 New field
//           confidence: d.confidence,
//           timestamp: d.timestamp,
//           trackingId: d.trackingId.toString(),
//           bbox: d.bbox,
//           imagePath: cleanPath,
//           screenshotUrl: `${BASE_URL}/${cleanPath}`,
//           processingTime: processingTimeStr,
//           videoUrl: videoUrl // Saved for reference
//         };
//       });
//       await Detection.insertMany(detectionsToSave);
//     }

//     return res.json({
//       message: "Success",
//       processing_time: processingTimeStr,
//       mode: imageFile ? "Image Search" : "Text Search",
//       counts: finalCounts,
//       totalUniqueObjects: new Set(detections.map((d) => d.trackingId)).size,
//       results: detections.map((d) => ({
//         ...d,
//         screenshotUrl: `${BASE_URL}/${d.image_path.replace(/\\/g, "/")}`,
//       })),
//     });

//   } catch (err) {
//     console.error("❌ ERROR:", err.message);
//     return res.status(500).json({ error: err.message });
//   }
// };

exports.uploadAndProcess = async (req, res) => {
  try {
    const videoFile = req.files?.file?.[0];
    const imageFile = req.files?.image?.[0]; 
    const userPrompt = req.body.text || "license plate";

    if (!videoFile) return res.status(400).json({ message: "Video file missing" });
    
    const startTime = Date.now();
    const videoUrl = `${process.env.BASE_URL}/uploads/${videoFile.filename}`;
    const imageUrl = imageFile ? `${process.env.BASE_URL}/uploads/${imageFile.filename}` : null;

    // 1. Python API Call
    const response = await axios.post(`${process.env.PYTHON_API_URL}/process`, {
        fileUrl: videoUrl,
        imageUrl: imageUrl,
        prompt: userPrompt,
      },
      { timeout: 0, maxContentLength: Infinity, maxBodyLength: Infinity }
    );

    const endTime = Date.now();
    const processingTimeStr = `${((endTime - startTime) / 1000).toFixed(2)}s`;

   const detections = (response.data.results || []).map(d => {

  const cleanPath = (d.image_path || "").replace(/\\/g, "/")

  return {
    object: d.object || "unknown",
    ocrText: d.ocr_text || "",
    confidence: typeof d.confidence === "number" ? d.confidence : 0.85,
    trackingId: (d.trackingId || Date.now()).toString(),
    timestamp: d.timestamp || "0",
    bbox: d.bbox || [],
    imagePath: cleanPath
  }

}).filter(d => d.confidence >= 0.30)

    const BASE_URL = process.env.BASE_URL;
    const finalCounts = {};

    // 3. Dynamic Keywords Counting (Object Name + OCR Text)
    const keywords = userPrompt.toLowerCase().split(/[\s,]+/)
      .filter(w => w.length > 2 && !["with", "and", "the"].includes(w));

    keywords.forEach((key) => {
      const uniqueSet = new Set();
      detections.forEach((d) => {
        const inObject = d.object.toLowerCase().includes(key);
        const inOCR = d.ocrText && d.ocrText.toLowerCase().includes(key);
        // const inOCR = d.ocr_text && d.ocr_text.toLowerCase().includes(key);
        
        if (inObject || inOCR) {
          uniqueSet.add(d.trackingId);
        }
      });
      finalCounts[`total_${key.replace(/[^a-zA-Z0-9]/g, '')}`] = uniqueSet.size;
    });

    // 4. Save to Database
    if (detections.length > 0) {
      const detectionsToSave = detections.map((d) => ({
          fileName: videoFile.filename,
          textNote: userPrompt,
          object: d.object,
          ocrText: d.ocr_text,
          confidence: d.confidence || 0.85,
          // confidence: d.confidence,
          timestamp: d.timestamp,
          trackingId: (d.trackingId || Date.now()).toString(),
          // trackingId: d.trackingId.toString(),
          bbox: d.bbox,
          imagePath: d.image_path.replace(/\\/g, "/"),
          screenshotUrl: `${BASE_URL}/${d.image_path.replace(/\\/g, "/")}`,
          processingTime: processingTimeStr,
          videoUrl: videoUrl
      }));
      await Detection.insertMany(detectionsToSave);
    }

    // 5. Final JSON Response
    return res.json({
      message: "Success",
      processing_time: processingTimeStr,
      mode: imageFile ? "Image Search" : "Text Search",
      counts: finalCounts,
      totalUniqueObjects: new Set(detections.map(d => d.trackingId)).size,
      results: detections.map((d) => ({
        ...d,
        screenshotUrl: `${BASE_URL}/${d.image_path.replace(/\\/g, "/")}`,
      })),
    });

  } catch (err) {
    console.error("❌ Node Error:", err.message);
    return res.status(500).json({ error: err.message });
  }
};