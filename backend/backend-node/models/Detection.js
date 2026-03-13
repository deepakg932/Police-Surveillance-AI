const mongoose = require("mongoose");

const detectionSchema = new mongoose.Schema({
  fileName: String,
  imageName: String,
  imagePath: String, // path to saved image
  textNote: String,
  videoUrl: String, // 🎯 Added this field

  object: String,
  ocrText: String, // 👈 Ye field zaroor add karein
  // upperColor: String, // color field for persons
  color: String,
  helmet: Boolean,
  vehicle: String,

  timestamp: String,
  confidence: Number,
  
  screenshotUrl: String,
  trackingId: {
    type: String,
    index: true
  },
  bbox: [Number],
  processingTime: String,

  createdAt: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model("Detection", detectionSchema);
