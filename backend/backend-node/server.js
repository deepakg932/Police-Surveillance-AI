// const express = require("express");
// const cors = require("cors");
// require("dotenv").config();
// const app = express();
// const connectDB = require("./config/db");
// app.use(cors());
// app.use(express.json());
// connectDB();

// const authRoute =  require("./routes/authRoute.js")
// const videoRoutes = require("./routes/videoRoute");


// app.use("/api", uploadRoute);




// app.use(express.json({ limit: "500mb" }));
// app.use(express.urlencoded({ limit: "500mb", extended: true }));
// app.use("/api-auth",authRoute)

// app.use("/api/video", videoRoutes);

// // test route
// app.get("/", (req, res) => {
//   res.send("Backend running 🚀");
// });

// app.listen(5000, () => console.log("Server running on port 5000"));


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

app.listen(5000, () => console.log("Server running on port 5000"));