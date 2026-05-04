const express = require("express");
const router = express.Router();
// const multer = require("multer");

const { uploadAndProcess, detectFromImage, searchDetections,   deleteHistoryEntry,
  deleteHistoryBatch,
  clearHistory,  getHistory, askVideoQuestion ,uploadAndProcessMultiple,    getBatchStatus,              // ✅ NEW
deleteByTrackingId,deleteMultiple,getJobStatus,downloadDetectionImage,markJobFailed} = require("../Controllers/videoController.js")
const authMiddleware = require("../middleware/authMiddleware.js");
const { multiUpload, multiVideoUpload } = require("../middleware/upload.js");





// 🔐 protected routes
router.post("/upload", authMiddleware, multiUpload, uploadAndProcess);
// Image-only detection (no video mixing)
router.post("/image", authMiddleware, multiUpload, detectFromImage);
router.get("/search", authMiddleware, searchDetections);
// Video + prompt - AI apne prompt ke hisaab se detect/answer karega
router.delete("/deleteall", authMiddleware, deleteMultiple);
router.delete("/delete/:trackingId", authMiddleware, deleteByTrackingId);
router.get("/status/:jobId", authMiddleware, getJobStatus);
router.post("/upload/multi", authMiddleware, multiVideoUpload, uploadAndProcessMultiple);
router.get("/batch/:batchId", authMiddleware, getBatchStatus);
router.get("/history", authMiddleware, getHistory);
router.delete("/history/clear", authMiddleware, clearHistory);
router.delete("/history/batch", authMiddleware, deleteHistoryBatch);
router.delete("/history/:entryId", authMiddleware, deleteHistoryEntry);
router.get("/download-image", downloadDetectionImage);
router.post("/status/:jobId/fail", markJobFailed);

module.exports = router;