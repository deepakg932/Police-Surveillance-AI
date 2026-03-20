const jwt = require("jsonwebtoken");
const { JWT_SECRET } = require("../config");

const authMiddleware = (req, res, next) => {
  try {
    // let token = req.headers.authorization;
        const token = req.headers.authorization?.split(" ")[1];

    console.log("Raw token:", token);

    if (!token) {
      return res.status(401).json({ message: "No token provided" });
    }

    // 🔥 Bearer remove karo
    if (token.startsWith("Bearer ")) {
      token = token.split(" ")[1];
    }

    const decoded = jwt.verify(token, JWT_SECRET);

    console.log("Decoded user:", decoded);
    req.userId = decoded.userId; 

    // req.userI = decoded;

    next();
  } catch (err) {
    console.error("Auth error:", err.message);
    return res.status(401).json({ message: "Invalid token" });
  }
};

module.exports = authMiddleware;