const multer = require("multer");
const path = require("path");
const fs = require("fs");

const uploadDir = path.join(__dirname, "..", "uploads");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const unique = `${Date.now()}_${Math.random().toString(36).slice(2)}`;
    const ext = path.extname(file.originalname);
    cb(null, `${unique}${ext}`);
  },
});

const fileFilter = (req, file, cb) => {
  const allowed = [
    "video/mp4", "video/avi", "video/mov", "video/mkv",
    "video/webm", "video/x-matroska",
    "image/jpeg", "image/png", "image/jpg", "image/webp",
  ];
  cb(null, allowed.includes(file.mimetype));
};

const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 2 * 1024 * 1024 * 1024 }, // 2GB
});

// ✅ Single video + optional image
exports.multiUpload = upload.fields([
  { name: "file", maxCount: 1 },
  { name: "image", maxCount: 1 },
]);

// ✅ Multiple videos + optional image
exports.multiVideoUpload = upload.fields([
  { name: "files", maxCount: 20 },
  { name: "image", maxCount: 1 },
]);