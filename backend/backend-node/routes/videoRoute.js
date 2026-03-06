const express = require("express");
const router = express.Router();
// const multer = require("multer");

const { uploadAndProcess, searchDetections } = require("../Controllers/videoController.js")
const authMiddleware = require("../middleware/authMiddleware.js");
const  multiUpload = require("../middleware/upload.js");

// const storage = multer.diskStorage({
//   destination: "uploads/",
//   filename: (req, file, cb) => {
//     cb(null, Date.now() + "-" + file.originalname);
//   },
// });

// const upload = multer({ storage });


// const upload = multer({
//   storage: storage,
//   limits: {
//     fileSize: 500 * 1024 * 1024, // 500MB max
//   },
// });




// 🔐 protected routes
router.post("/upload", authMiddleware, multiUpload, uploadAndProcess);
router.get("/search", authMiddleware, searchDetections);

module.exports = router;