


const axios = require("axios");
const Detection = require("../models/Detection.js");
const path = require("path")

exports.searchDetections = async (req, res) => {
  try {
    const { object, color, helmet, vehicle } = req.query;

    let filter = {};

    if (object) filter.object = object;
    if (color) filter.color = color;
    if (helmet) filter.helmet = helmet === "true";
    if (vehicle) filter.vehicle = vehicle;

    const results = await Detection.find(filter);

    res.json(results);
  } catch (err) {
    res.status(500).json({ error: err.message });
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


exports.uploadAndProcess = async (req, res) => {
  try {
    const videoFile = req.files?.file?.[0];
    console.log(videoFile,"ppppppppppp")
    const imageFile = req.files?.image?.[0]; // Optional Image
    console.log(imageFile,"mmmmmmmmmmm")
    const userPrompt = req.body.text || "person, car"; 
    console.log(userPrompt,"oooooooooooo")

    if (!videoFile) return res.status(400).json({ message: "Video missing" });

    const videoPath = path.resolve(videoFile.path);
    const imagePath = imageFile ? path.resolve(imageFile.path) : null;

    // Python API Call
    const response = await axios.post("http://127.0.0.1:8000/process", {
      filePath: videoPath,
      imagePath: imagePath,
      prompt: userPrompt,
    });

    // const detections = response.data.results || [];
    // Node.js Controller ke andar detections filter kar lo:
const detections = (response.data.results || []).filter(d => d.confidence >= 0.60);

    // 🎯 DYNAMIC MULTI-COUNT LOGIC
    const finalCounts = {};
    // Prompt ko split karo (e.g., "person with helmet" -> ["person", "helmet"])
    const keywords = userPrompt.toLowerCase().split(/[\s,]+/)
      .filter(w => !["with", "and", "wearing", "in", "a"].includes(w));

    keywords.forEach(key => {
      // Har keyword ke liye unique trackingId dhoondo
      const uniqueSet = new Set();
      detections.forEach(d => {
        if (d.object.toLowerCase().includes(key)) {
          uniqueSet.add(d.trackingId);
        }
      });
      finalCounts[`total_${key}`] = uniqueSet.size;
    });

    // Database Bulk Save
    if (detections.length > 0) {
      await Detection.insertMany(detections.map(d => ({
        fileName: videoFile.filename,
        textNote: userPrompt,
        object: d.object,
        confidence: d.confidence,
        timestamp: d.timestamp,
        trackingId: d.trackingId,
        bbox: d.bbox,
        image_path: d.image_path
      })));
    }

    return res.json({
      message: "Success",
      mode: imageFile ? "Image Search" : "Text Search",
      counts: finalCounts, // e.g., { total_person: 5, total_helmet: 5 }
      totalUniqueObjects: new Set(detections.map(d => d.trackingId)).size,
      results: detections.map(d => ({
        ...d,
        screenshotUrl: `http://localhost:5000/${d.image_path}`
      }))
    });

  } catch (err) {
    console.error("❌ ERROR:", err.message);
    return res.status(500).json({ error: err.message });
  }
};