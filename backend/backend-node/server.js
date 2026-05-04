
const express = require("express");
const cors = require("cors");
require("dotenv").config();
const path = require("path")

const app = express();
const connectDB = require("./config/db");

connectDB();
app.use(express.json());
app.use(cors());

app.use("/detected_frames",
express.static(path.join(__dirname,"../ai-python/detected_frames")));
app.use(express.json({ limit: "500mb" }));
app.use(express.urlencoded({ limit: "500mb", extended: true }));
const axios = require("axios");

// routes
const authRoute = require("./routes/authRoute.js");
const videoRoutes = require("./routes/videoRoute");

const filesDir = process.env.FILES_DIR || "public/files";
app.use("/files", express.static(filesDir));
// auth routes
app.use("/api-auth", authRoute);
app.use("/results", express.static(path.join(__dirname, "../ai-python/results")));

app.use("/uploads", express.static(path.join(__dirname, "uploads")));
console.log(__dirname)
// video routes
app.use("/api/video", videoRoutes);
app.use("/uploads", express.static("uploads"));



async function cancelActivePythonJobs() {
  const jobs = Array.from(global.activePythonJobs || []);

  console.log("Active Python jobs to cancel:", jobs);

  for (const jobId of jobs) {
    try {
      await axios.post(
        `${process.env.PYTHON_API_URL}/process/cancel/${jobId}`,
        {},
        { timeout: 10000 }
      );
      console.log("✅ Cancelled Python job:", jobId);
    } catch (err) {
      console.error("❌ Cancel failed:", jobId, err.message);
    }
  }
}

let shuttingDown = false;

async function gracefulShutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;

  console.log(`${signal} received. Cancelling Python jobs...`);
  await cancelActivePythonJobs();

  setTimeout(() => process.exit(0), 1000);
}

process.once("SIGINT", () => gracefulShutdown("SIGINT"));
process.once("SIGTERM", () => gracefulShutdown("SIGTERM"));

// test
app.get("/", (req, res) => {
  res.send("Backend running 🚀");
});

app.listen(5000, () => console.log("Server running on port 5000"));