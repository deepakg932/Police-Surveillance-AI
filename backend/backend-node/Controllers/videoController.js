// const axios = require("axios");
// const Detection = require("../models/Detection.js");
// const path = require("path");

// exports.searchDetections = async (req, res) => {
//   try {
//     const { object, textNote, trackingId, fileName } = req.query;
//     let filter = {};

//     // 1. Filter Setup (Regex for dynamic search)
//     if (object) filter.object = { $regex: object, $options: "i" };
//     if (textNote) filter.textNote = { $regex: textNote, $options: "i" };
//     if (trackingId) filter.trackingId = trackingId;
//     if (fileName) filter.fileName = fileName;

//     // 2. Database se results nikaalo
//     const results = await Detection.find(filter).sort({ createdAt: -1 });

//     if (results.length === 0) {
//       return res.json({
//         message: "No results found",
//         counts: {},
//         totalUniqueObjects: 0,
//         results: [],
//       });
//     }

//     const finalCounts = {};

//     const searchTerms = (object || results[0].textNote || "")
//       .toLowerCase()
//       .split(/[\s,]+/)
//       .filter(
//         (w) => !["with", "and", "wearing", "in", "a", "wear"].includes(w),
//       );

//     searchTerms.forEach((key) => {
//       const uniqueSetForTerm = new Set();
//       results.forEach((d) => {
//         if (d.object.toLowerCase().includes(key)) {
//           uniqueSetForTerm.add(d.trackingId);
//         }
//       });
//       finalCounts[`total_${key}`] = uniqueSetForTerm.size;
//     });

//     return res.json({
//       message: "Success",
//       mode: "Search Result",
//       counts: finalCounts,
//       totalUniqueObjects: new Set(results.map((d) => d.trackingId)).size,
//       results: results.map((d) => ({
//         object: d.object,
//         confidence: d.confidence,
//         trackingId: d.trackingId,
//         timestamp: d.timestamp,
//         image_path: d.imagePath, // DB field name match karo
//         bbox: d.bbox,
//         processing_time: d.processingTime,
//         screenshotUrl: d.screenshotUrl, // DB se direct URL
//       })),
//     });
//   } catch (err) {
//     console.error("❌ Search Error:", err.message);
//     return res.status(500).json({ error: err.message });
//   }
// };

// //yh wali api sahi h abhi chl ri h
// // exports.uploadAndProcess = async (req, res) => {
// //   try {
// //     const videoFile = req.files?.file?.[0];
// //     console.log(videoFile,"kkkkkkkkkkkkkkkk")
// //     // User input: e.g., "blue car, person with helmet, dog"
// //     const userPrompt = req.body.text || "person, vehicle, helmet";
// //     console.log(userPrompt,"jjjjjjjjjjj")

// //     if (!videoFile) return res.status(400).json({ message: "Video missing" });

// //     const filePath = path.resolve(videoFile.path);

// //     // Python call with prompt
// //     const response = await axios.post("http://127.0.0.1:8000/process", {
// //       filePath: filePath,
// //       prompt: userPrompt,
// //     });

// //     const detections = response.data.results || [];

// //     // Save to MongoDB
// //     const savedData = await Detection.insertMany(
// //       detections.map((d) => ({
// //         fileName: videoFile.filename,
// //         textNote: userPrompt,
// //         object: d.object,
// //         confidence: d.confidence,
// //         timestamp: d.timestamp,
// //         trackingId: d.trackingId,
// //         bbox: d.bbox
// //       }))
// //     );

// //     return res.json({
// //       message: "Success",
// //       totalObjects: new Set(detections.map(d => d.trackingId)).size,
// //       results: detections,
// //     });

// //   } catch (err) {
// //     console.error("❌ ERROR:", err.message);
// //     return res.status(500).json({ error: err.message });
// //   }
// // };

// exports.uploadAndProcess = async (req, res) => {
//   try {
//     const videoFile = req.files?.file?.[0];
//     console.log(videoFile, "ppppppppppp");
//     const imageFile = req.files?.image?.[0]; // Optional Image
//     console.log(imageFile, "mmmmmmmmmmm");
//     const userPrompt = req.body.text || "person, car";
//     console.log(userPrompt, "oooooooooooo");

//     if (!videoFile) return res.status(400).json({ message: "Video missing" });
//     const startTime = Date.now();
//     console.log(startTime, "startt");

//     const videoPath = path.resolve(videoFile.path);
//     const imagePath = imageFile ? path.resolve(imageFile.path) : null;

//     const response = await axios.post("http://127.0.0.1:8000/process", {
//       filePath: videoPath,
//       imagePath: imagePath,
//       prompt: userPrompt,
//     });

//     // const detections = response.data.results || [];
//     const endTime = Date.now();
//     console.log(endTime, "hhhhhh");

//     const durationSeconds = ((endTime - startTime) / 1000).toFixed(2);
//     console.log(durationSeconds, "startkkk");
//     const processingTimeStr = `${durationSeconds}s`;
//     console.log(processingTimeStr, "ssssssss");

//     const detections = (response.data.results || []).filter(
//       (d) => d.confidence >= 0.35,
//     );
//     const BASE_URL = "http://localhost:5000";

//     const finalCounts = {};

//     const keywords = userPrompt
//       .toLowerCase()
//       .split(/[\s,]+/)
//       .filter((w) => !["with", "and", "wearing", "in", "a"].includes(w));

//     keywords.forEach((key) => {
//       const uniqueSet = new Set();
//       detections.forEach((d) => {
//         if (d.object.toLowerCase().includes(key)) {
//           uniqueSet.add(d.trackingId);
//         }
//       });
//       finalCounts[`total_${key}`] = uniqueSet.size;
//     });

//     // if (detections.length > 0) {
//     //   await Detection.insertMany(detections.map(d => (
//     //     {
//     //     fileName: videoFile.filename,
//     //     textNote: userPrompt,
//     //     object: d.object,
//     //     confidence: d.confidence,
//     //     timestamp: d.timestamp,
//     //     trackingId: d.trackingId,
//     //     bbox: d.bbox,
//     //     image_path: d.image_path
//     //   })));
//     // }
//     if (detections.length > 0) {
//       await Detection.insertMany(
//         detections.map((d) => {
//           const cleanPath = d.image_path.replace(/\\/g, "/");

//           return {
//             fileName: videoFile.filename,
//             textNote: userPrompt,
//             object: d.object,
//             confidence: d.confidence,
//             timestamp: d.timestamp,
//             trackingId: d.trackingId.toString(), // String mein convert karna safe hai
//             bbox: d.bbox,
//             imagePath: cleanPath, // DB field: imagePath
//             screenshotUrl: `${BASE_URL}/${cleanPath}`, // 🎯 DB field: screenshotUrl (AB SAVE HOGA)
//             processingTime: processingTimeStr,
//           };
//         }),
//       );
      
//     }
//     return res.json({
//       message: "Success",
//       processing_time: processingTimeStr,
//       mode: imageFile ? "Image Search" : "Text Search",
//       counts: finalCounts, // e.g., { total_person: 5, total_helmet: 5 }
//       totalUniqueObjects: new Set(detections.map((d) => d.trackingId)).size,
//       // results: detections.map(d => ({
//       //   ...d,
//       //   screenshotUrl: `http://localhost:5000/${d.image_path}`
//       // }))
//       results: detections.map((d) => {
//         const cleanPath = d.image_path.replace(/\\/g, "/");
//         return {
//           ...d,
//           screenshotUrl: `${BASE_URL}/${cleanPath}`,
//         };
//       }),
//     });
//   } catch (err) {
//     console.error("❌ ERROR:", err.message);
//     return res.status(500).json({ error: err.message });
//   }
// };



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


//yh wali api sahi h abhi chl ri h
// exports.uploadAndProcess = async (req, res) => {
//   try {
//     const videoFile = req.files?.file?.[0];
//     console.log(videoFile,"kkkkkkkkkkkkkkkk")
//     // User input: e.g., "blue car, person with helmet, dog"
//     const userPrompt = req.body.text || "person, vehicle, helmet";
//     console.log(userPrompt,"jjjjjjjjjjj")

//     if (!videoFile) return res.status(400).json({ message: "Video missing" });

//     const filePath = path.resolve(videoFile.path);

//     // Python call with prompt
//     const response = await axios.post("http://127.0.0.1:8000/process", {
//       filePath: filePath,
//       prompt: userPrompt,
//     });

//     const detections = response.data.results || [];

//     // Save to MongoDB
//     const savedData = await Detection.insertMany(
//       detections.map((d) => ({
//         fileName: videoFile.filename,
//         textNote: userPrompt,
//         object: d.object,
//         confidence: d.confidence,
//         timestamp: d.timestamp,
//         trackingId: d.trackingId,
//         bbox: d.bbox
//       }))
//     );

//     return res.json({
//       message: "Success",
//       totalObjects: new Set(detections.map(d => d.trackingId)).size,
//       results: detections,
//     });

//   } catch (err) {
//     console.error("❌ ERROR:", err.message);
//     return res.status(500).json({ error: err.message });
//   }
// };



















//2nd api sahi bhaii
// exports.uploadAndProcess = async (req, res) => {
//   try {
//     const videoFile = req.files?.file?.[0];
//     console.log(videoFile, "ppppppppppp");
//     const imageFile = req.files?.image?.[0]; // Optional Image
//     console.log(imageFile, "mmmmmmmmmmm");
//     const userPrompt = req.body.text || "person, car";
//     console.log(userPrompt, "oooooooooooo");

//     if (!videoFile) return res.status(400).json({ message: "Video missing" });
//     const startTime = Date.now();
//     console.log(startTime, "startt");

//     const videoPath = path.resolve(videoFile.path);
//     const imagePath = imageFile ? path.resolve(imageFile.path) : null;

//     const response = await axios.post("http://127.0.0.1:8000/process", {
//       filePath: videoPath,
//       imagePath: imagePath,
//       prompt: userPrompt,
//    }, { timeout: 600000 }); //

//     // const detections = response.data.results || [];
//     const endTime = Date.now();
//     console.log(endTime, "hhhhhh");

//     const durationSeconds = ((endTime - startTime) / 1000).toFixed(2);
//     console.log(durationSeconds, "startkkk");
//     const processingTimeStr = `${durationSeconds}s`;
//     console.log(processingTimeStr, "ssssssss");

//     const detections = (response.data.results || []).filter(
//       (d) => d.confidence >= 0.6,
//     );
//     // const BASE_URL = "http://localhost:5000";
//     const BASE_URL = "https://75w2s5b8-5000.inc1.devtunnels.ms"
//     console.log(BASE_URL,"kkkk")

//     const finalCounts = {};

//     const keywords = userPrompt
//       .toLowerCase()
//       .split(/[\s,]+/)
//       .filter((w) => !["with", "and", "wearing", "in", "a"].includes(w));

//     keywords.forEach((key) => {
//       const uniqueSet = new Set();
//       detections.forEach((d) => {
//         if (d.object.toLowerCase().includes(key)) {
//           uniqueSet.add(d.trackingId);
//         }
//       });
//       finalCounts[`total_${key}`] = uniqueSet.size;
//     });

//     // if (detections.length > 0) {
//     //   await Detection.insertMany(detections.map(d => (
//     //     {
//     //     fileName: videoFile.filename,
//     //     textNote: userPrompt,
//     //     object: d.object,
//     //     confidence: d.confidence,
//     //     timestamp: d.timestamp,
//     //     trackingId: d.trackingId,
//     //     bbox: d.bbox,
//     //     image_path: d.image_path
//     //   })));
//     // }
//     if (detections.length > 0) {
//       await Detection.insertMany(
//         detections.map((d) => {
//           const cleanPath = d.image_path.replace(/\\/g, "/");

//           return {
//             fileName: videoFile.filename,
//             textNote: userPrompt,
//             object: d.object,
//             confidence: d.confidence,
//             timestamp: d.timestamp,
//             trackingId: d.trackingId.toString(), // String mein convert karna safe hai
//             bbox: d.bbox,
//             imagePath: cleanPath, // DB field: imagePath
//             screenshotUrl: `${BASE_URL}/${cleanPath}`, // 🎯 DB field: screenshotUrl (AB SAVE HOGA)
//             processingTime: processingTimeStr,
//           };
//         }),
//       );
     
//     }
//     return res.json({
//       message: "Success",
//       processing_time: processingTimeStr,
//       mode: imageFile ? "Image Search" : "Text Search",
//       counts: finalCounts, // e.g., { total_person: 5, total_helmet: 5 }
//       totalUniqueObjects: new Set(detections.map((d) => d.trackingId)).size,
//       // results: detections.map(d => ({
//       //   ...d,
//       //   screenshotUrl: `http://localhost:5000/${d.image_path}`
//       // }))
//       results: detections.map((d) => {
//         const cleanPath = d.image_path.replace(/\\/g, "/");
//         return {
//           ...d,
//           screenshotUrl: `${BASE_URL}/${cleanPath}`,
//         };
//       }),
//     });
//   } catch (err) {
//     console.error("❌ ERROR:", err.message);
//     return res.status(500).json({ error: err.message });
//   }
// };





// exports.uploadAndProcess = async (req, res) => {
//   try {
//     const videoFile = req.files?.file?.[0];
//     const userPrompt = req.body.text || "helmet";
//     const BASE_URL = "http://localhost:5000";

//     const response = await axios.post("http://127.0.0.1:8000/process", {
//       filePath: path.resolve(videoFile.path),
//       prompt: userPrompt,
//     });

//     // 🎯 FILTER: 0.50 ya 0.60 confidence (Aapki choice)
//     const detections = (response.data.results || []).filter(d => d.confidence >= 0.5);

//     const finalCounts = {};
//     const promptList = userPrompt.split(',').map(p => p.trim().toLowerCase());

//     promptList.forEach((phrase) => {
//       const countsPerFrame = {}; 
//       detections.forEach((d) => {
//         if (d.object.toLowerCase().includes(phrase)) {
//           countsPerFrame[d.timestamp] = (countsPerFrame[d.timestamp] || 0) + 1;
//         }
//       });
//       // Har frame mein max kitne dikhe
//       finalCounts[`total_${phrase.replace(/\s+/g, '_')}`] = Object.values(countsPerFrame).length > 0 ? Math.max(...Object.values(countsPerFrame)) : 0;
//     });

//     // Sirf images wale results dikhayein (Unique Screenshots)
//     const uniqueDetections = detections.filter(d => d.image_path !== "");

//     return res.json({
//       message: "Success",
//       counts: finalCounts,
//       totalUniqueObjects: Object.values(finalCounts).reduce((a, b) => a + b, 0),
//       results: uniqueDetections.map(d => ({
//         ...d,
//         screenshotUrl: `${BASE_URL}/${d.image_path.replace(/\\/g, "/")}`
//       }))
//     });
//   } catch (err) {
//     res.status(500).json({ error: err.message });
//   }
// };



exports.uploadAndProcess = async (req, res) => {
  try {
    const videoFile = req.files?.file?.[0];
    const imageFile = req.files?.image?.[0]; 
    const userPrompt = req.body.text || "person, car";

    if (!videoFile) return res.status(400).json({ message: "Video missing" });
    
    const startTime = Date.now();
    const videoPath = path.resolve(videoFile.path);
    const imagePath = imageFile ? path.resolve(imageFile.path) : null;
    const BASE_URL = process.env.BASE_URL;

    // 🎯 Video URL Construct karein
    const videoUrl = `${BASE_URL}/uploads/${videoFile.filename}`;

    const response = await axios.post(`${process.env.PYTHON_API_URL}/process`, {
      filePath: videoPath,
      imagePath: imagePath,
      prompt: userPrompt,
    });

    const endTime = Date.now();
    const durationSeconds = ((endTime - startTime) / 1000).toFixed(2);
    const processingTimeStr = `${durationSeconds}s`;

    const detections = (response.data.results || []).filter(
      (d) => d.confidence >= 0.20
    );

    // ... (Aapka existing keyword counting aur DB insertion logic yahan rahega) ...

    return res.json({
      message: "Success",
      videoUrl: videoUrl, // 🚀 Response mein Video URL add kiya
      processing_time: processingTimeStr,
      mode: imageFile ? "Image Search" : "Text Search",
      counts: finalCounts,
      totalUniqueObjects: new Set(detections.map((d) => d.trackingId)).size,
      results: detections.map((d) => {
        const cleanPath = d.image_path.replace(/\\/g, "/");
        return {
          ...d,
          screenshotUrl: `${BASE_URL}/${cleanPath}`,
        };
      }),
    });
  } catch (err) {
    console.error("❌ ERROR:", err.message);
    return res.status(500).json({ error: err.message });
  }
};
// ============ MULTIMODAL AI - Prompt-based Video Understanding ============
// Video + jo bhi prompt do, AI uske according detect/answer karega
exports.askVideoQuestion = async (req, res) => {
  try {
    const videoFile = req.file || req.files?.file?.[0];
    const prompt = req.body.prompt || req.body.text || "Video mein kya kya dikh raha hai batao.";

    if (!videoFile) return res.status(400).json({ message: "Video file required" });

    const videoPath = path.resolve(videoFile.path);
    const BASE_URL = process.env.BASE_URL;
    
    // 🎯 Video URL Construct karein
    const videoUrl = `${BASE_URL}/uploads/${videoFile.filename}`;

    const response = await axios.post(
      `${process.env.PYTHON_API_URL}/ask`,
      { videoPath, prompt },
      { timeout: 300000 } 
    );

    return res.json({
      message: "Success",
      videoUrl: videoUrl, // 🚀 Yahan bhi Video URL add kiya
      answer: response.data.answer,
      frameCount: response.data.frameCount,
    });
  } catch (err) {
    console.error("❌ ASK Error:", err.message);
    return res.status(500).json({ error: err.message });
  }
};