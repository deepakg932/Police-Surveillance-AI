const express = require("express");
const router = express.Router();
// const multer = require("multer");

const {
  uploadAndProcess,
  detectFromImage,
  searchDetections,
  askVideoQuestion,
  deleteByTrackingId,
  deleteMultiple,
  getJobStatus,
  getHistory,
  deleteHistoryEntry,
  deleteHistoryBatch,
  clearHistory,
} = require("../Controllers/videoController.js");
const authMiddleware = require("../middleware/authMiddleware.js");
const { multiUpload} = require("../middleware/upload.js");





// 🔐 protected routes
router.post("/upload", authMiddleware, multiUpload, uploadAndProcess);
router.delete("/delete/:trackingId", authMiddleware, deleteByTrackingId);
// Image-only detection (no video mixing)
router.post("/image", authMiddleware, multiUpload, detectFromImage);
router.get("/search", authMiddleware, searchDetections);
// Video + prompt - AI apne prompt ke hisaab se detect/answer karega
router.delete("/deleteall", authMiddleware, deleteMultiple);
router.get("/status/:jobId", authMiddleware, getJobStatus);
router.get("/history", authMiddleware, getHistory);
router.delete("/history/batch", authMiddleware, deleteHistoryBatch);
router.delete("/history/:entryId", authMiddleware, deleteHistoryEntry);
router.delete("/history", authMiddleware, clearHistory);

module.exports = router;