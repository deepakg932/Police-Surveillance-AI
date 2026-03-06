// const axios = require("axios");
// const Detection = require("../models/Detection.js");

// exports.uploadAndProcess = async (req, res) => {
//   try {
//     const filePath = req.file.path;

//     // 🔥 call Python AI service
//     const response = await axios.post("http://127.0.0.1:8000/process", {
//       filePath,
//     });

//     const detections = response.data.results;

//     // 💾 save in DB
//     const saved = await Detection.insertMany(
//       detections.map((d) => ({
//         fileName: req.file.filename,
//         object: d.object,
//         color: d.color || "",
//         helmet: d.helmet || false,
//         vehicle: d.vehicle || "",
//         timestamp: d.timestamp,
//         confidence: d.confidence,
//         trackingId: d.trackingId || 0,
//       }))

      
//     );

//     console.log(saved,"saved")

//    return res.json({
//       message: "Video processed successfully",
//       results: saved,
//     });
//   } catch (err) {
//     console.error("PROCESS ERROR:", err.message);
//     res.status(500).json({ error: err.message });
//   }
// };


// exports.searchDetections = async (req, res) => {
//   try {
//     const { object, color, helmet, vehicle } = req.query;

//     let filter = {};

//     if (object) filter.object = object;
//     if (color) filter.color = color;
//     if (helmet) filter.helmet = helmet === "true";
//     if (vehicle) filter.vehicle = vehicle;

//     const results = await Detection.find(filter);

//     res.json(results);
//   } catch (err) {
//     res.status(500).json({ error: err.message });
//   }
// };


const axios = require("axios");
const Detection = require("../models/Detection.js");
const path = require("path")
// exports.uploadAndProcess = async (req, res) => {
//   try {
//     const videoFile = req.files?.file?.[0];
//     const imageFile = req.files?.image?.[0];
//     const textData = req.body.text || "";

//     if (!videoFile) {
//       return res.status(400).json({ message: "Video file is required" });
//     }

//     const filePath = path.resolve(videoFile.path);

//     console.log("📹 VIDEO:", videoFile.filename);
//     console.log("📝 TEXT:", textData);
//     console.log("📁 PATH:", filePath);

//     // 🔥 Python call
//     const response = await axios.post("http://127.0.0.1:8000/process", {
//       filePath,
//     });

//     const detections = response.data.results || [];

//     // 🎨 color extract
//     function extractColor(text) {
//       text = text.toLowerCase();
//       if (text.includes("white")) return "white";
//       if (text.includes("blue")) return "blue";
//       if (text.includes("red")) return "red";
//       if (text.includes("green")) return "green";
//       if (text.includes("black")) return "black";
//       return "";
//     }

//     const colorQuery = extractColor(textData);

//     // 🎯 filter
//     const filtered = detections.filter((d) => {
//       if (!colorQuery) return true;
//       return d.color === colorQuery;
//     });

//     // 🧠 UNIQUE PERSONS
//     const uniquePersons = new Set(filtered.map((d) => d.trackingId));
//     const totalPersons = uniquePersons.size;

//     // 🪖 UNIQUE HELMET PERSONS (🔥 MAIN FIX)
//     const uniqueHelmetPersons = new Set();

//     detections.forEach((d) => {
//       if (d.helmet === true) {
//         uniqueHelmetPersons.add(d.trackingId);
//       }
//     });

//     const totalHelmet = uniqueHelmetPersons.size;

//     const helmetOnlyResults = filtered.filter(d => d.helmet === true);

//     console.log("👨‍👨‍👦 PERSONS:", totalPersons);
//     console.log("🪖 HELMET PERSONS:", totalHelmet);

//     // 💾 save DB
//     const saved = await Detection.insertMany(
//       filtered.map((d) => ({
//         fileName: videoFile.filename,
//         imageName: imageFile ? imageFile.filename : "",
//         textNote: textData,
//         object: d.object,
//         color: d.color,
//         helmet: d.helmet,
//         vehicle: d.vehicle,
//         timestamp: d.timestamp,
//         confidence: d.confidence,
//         trackingId: d.trackingId,
//       }))
//     );

//     return res.json({
//       message: "Video processed successfully",
//       totalDetections: detections.length,
//       totalFiltered: filtered.length,
//       totalPersons,
//       totalHelmet,
//       results: helmetOnlyResults,
//     });
//   } catch (err) {
//     console.error("❌ PROCESS ERROR:", err.message);
//     return res.status(500).json({ error: err.message });
//   }
// };

// 🔍 SEARCH API
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




// exports.uploadAndProcess = async (req, res) => {
//   try {
//     const BASE_URL = "http://127.0.0.1:5000";
//     const videoFile = req.files?.file?.[0];
//     const imageFile = req.files?.image?.[0];
//     const textData = req.body.text || "";

//     if (!videoFile) {
//       return res.status(400).json({ message: "Video file is required" });
//     }

//     const filePath = path.resolve(videoFile.path);

//     // 🔥 Python call
//     const response = await axios.post("http://127.0.0.1:8000/process", {
//       filePath,
//     });

//     const rawDetections = response.data.results || [];

//     // 📸 attach screenshot URL
//     const detections = rawDetections.map((d) => ({
//       ...d,
//       screenshotUrl: d.image
//         ? `${BASE_URL}/results/${d.image}`
//         : "",
//     }));

//     // 🎥 video & image url
//     const videoUrl = `${BASE_URL}/uploads/${videoFile.filename}`;
//     const imageUrl = imageFile
//       ? `${BASE_URL}/uploads/${imageFile.filename}`
//       : "";

//     // 💾 save DB
//     await Detection.insertMany(
//       detections.map((d) => ({
//         fileName: videoFile.filename,
//         imageName: imageFile ? imageFile.filename : "",
//         textNote: textData,
//         object: d.object,
//         upperColor: d.upperColor,
//         helmet: d.helmet,
//         vehicle: d.vehicle,
//         timestamp: d.timestamp,
//         confidence: d.confidence,
//         trackingId: d.trackingId,
//         screenshot: d.image,
//       }))
//     );

//     return res.json({
//       message: "Video processed successfully",

//       videoUrl,
//       imageUrl,

//       total: detections.length,
//       results: detections,
//     });
//   } catch (err) {
//     console.error("❌ ERROR:", err.message);
//     return res.status(500).json({ error: err.message });
//   }
// }


exports.uploadAndProcess = async (req, res) => {
  try {
    const videoFile = req.files?.file?.[0];
    console.log(videoFile,"kkkkkkkkkkkkkkkk")
    // User input: e.g., "blue car, person with helmet, dog"
    const userPrompt = req.body.text || "person, vehicle, helmet"; 
    console.log(userPrompt,"jjjjjjjjjjj")

    if (!videoFile) return res.status(400).json({ message: "Video missing" });

    const filePath = path.resolve(videoFile.path);

    // Python call with prompt
    const response = await axios.post("http://127.0.0.1:8000/process", {
      filePath: filePath,
      prompt: userPrompt, 
    });

    const detections = response.data.results || [];

    // Save to MongoDB
    const savedData = await Detection.insertMany(
      detections.map((d) => ({
        fileName: videoFile.filename,
        textNote: userPrompt,
        object: d.object,
        confidence: d.confidence,
        timestamp: d.timestamp,
        trackingId: d.trackingId,
        bbox: d.bbox
      }))
    );

    return res.json({
      message: "Success",
      totalObjects: new Set(detections.map(d => d.trackingId)).size,
      results: detections,
    });

  } catch (err) {
    console.error("❌ ERROR:", err.message);
    return res.status(500).json({ error: err.message });
  }
};