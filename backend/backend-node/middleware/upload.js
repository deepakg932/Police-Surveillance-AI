const multer = require("multer");

const storage = multer.diskStorage({
  destination: "uploads/",
  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname);
  },
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 500 * 1024 * 1024,
  },
});

const multiUpload = upload.fields([
  { name: "file", maxCount: 5 },
  { name: "image", maxCount: 1 },
]);

// Sirf video - /ask route ke liye
const singleVideoUpload = upload.single("file");

module.exports = { multiUpload, singleVideoUpload };