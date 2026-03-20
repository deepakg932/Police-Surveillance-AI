const express = require("express");
const router = express.Router();
// const multer = require("multer");

const { uploadAndProcess, searchDetections, askVideoQuestion } = require("../Controllers/videoController.js");
const authMiddleware = require("../middleware/authMiddleware.js");
const { multiUpload} = require("../middleware/upload.js");





// 🔐 protected routes
router.post("/upload", authMiddleware, multiUpload, uploadAndProcess);
router.get("/search", authMiddleware, searchDetections);
// Video + prompt - AI apne prompt ke hisaab se detect/answer karega


module.exports = router;