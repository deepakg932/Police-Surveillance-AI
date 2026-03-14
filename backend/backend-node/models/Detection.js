const mongoose = require("mongoose");

const detectionSchema = new mongoose.Schema({

  fileName: String,
  imageName: String,

  imagePath: String,
  screenshotUrl: String,

  videoUrl: String,

  textNote: String,

  object: String,
  vehicle: String,
  color: String,
  helmet: Boolean,

  ocrText: String,

  confidence: Number,

  trackingId: {
    type: String,
    index: true
  },

  bbox: [Number],

  timestamp: Number,

  processingTime: String,

  createdAt: {
    type: Date,
    default: Date.now
  }

});

module.exports = mongoose.model("Detection", detectionSchema);