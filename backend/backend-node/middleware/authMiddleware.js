const jwt = require("jsonwebtoken");
const { JWT_SECRET } = require("../config");

const authMiddleware = (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader) {
      return res.status(401).json({ message: "No token provided" });
    }

    const token = authHeader.split(" ")[1]; // ✅ only once

    const decoded = jwt.verify(token, JWT_SECRET);

    console.log("Decoded:", decoded);

    // 🔥 IMPORTANT: check this
    req.userId = decoded.userId || decoded.id; 

    next();
  } catch (err) {
    console.error("Auth error:", err.message);
    return res.status(401).json({ message: "Invalid token" });
  }
};

module.exports = authMiddleware;