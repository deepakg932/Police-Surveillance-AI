

const express = require("express");
const cors = require("cors");
require("dotenv").config();
const path = require("path")

const app = express();
const connectDB = require("./config/db");


connectDB();

app.use(cors());

app.use("/detected_frames",
express.static(path.join(__dirname,"../ai-python/detected_frames")));
app.use(express.json({ limit: "500mb" }));
app.use(express.urlencoded({ limit: "500mb", extended: true }));

// routes
const authRoute = require("./routes/authRoute.js");
const videoRoutes = require("./routes/videoRoute");
app.use(cors())

// auth routes
app.use("/api-auth", authRoute);
app.use("/results", express.static(path.join(__dirname, "../ai-python/results")));

app.use("/uploads", express.static(path.join(__dirname, "uploads")));
// video routes
app.use("/api/video", videoRoutes);

// test
app.get("/", (req, res) => {
  res.send("Backend running 🚀");
});


let port = process.env.PORT || 5000


const server = app.listen(port, () =>
  console.log("Server running on port 5000")
);

// important
server.timeout = 10 * 60 * 1000; // 10 minutes
server.keepAliveTimeout = 10 * 60 * 1000;
server.headersTimeout = 10 * 60 * 1000;
// console.log(port,"kkk")
// app.listen(port, () => console.log("Server running on port 5000"));