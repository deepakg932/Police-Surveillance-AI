const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const User = require("../models/User");
const { JWT_SECRET } = require("../config");

const register = async (req, res) => {
  try {
    const {
      fullName,
      email,
      password,
      badgeNumber,
      department,
      phone,
      agreeTerms,
    } = req.body;
console.log(req.body,'req,body')

    // if (!agreeTerms) {
    //   return res
    //     .status(400)
    //     .json({ message: "You must agree to terms & conditions" });
    // }


    const existingEmail = await User.findOne({ email });
    if (existingEmail) {
      return res.status(400).json({ message: "Email already registered" });
    }


    const existingBadge = await User.findOne({ badgeNumber });
    if (existingBadge) {
      return res.status(400).json({ message: "Badge number already used" });
    }
console.log(existingBadge,"existingBadge")
   
    const hashedPassword = await bcrypt.hash(password, 10);
    console.log(hashedPassword,"hashedPassword")

    const user = await User.create({
      fullName,
      email,
      password: hashedPassword,
      badgeNumber,
      department,
      phone,
    });
    console.log(user,"user")

    res.status(201).json({
      message: "Officer registered successfully",
      userId: user._id,
      data:user
    });
  } catch (err) {
    console.error("Register Error:", err.message);
    res.status(500).json({ error: err.message });
  }
};


const login = async (req, res) => {
  try {
    const { email, password } = req.body;


    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ message: "Invalid email" });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ message: "Invalid password" });
    }

    
    const token = jwt.sign(
      {
        id: user._id,
        role: user.role,
      },
      JWT_SECRET,
      { expiresIn: "7d" }
    );

    return res.json({
      message: "Login successful",
      
      user: {
        // id: user._id,
        // fullName: user.fullName,
        email: user.email,
        department: user.department,
        token
      },
    });
  } catch (err) {
    console.error("Login Error:", err.message);
    return res.status(500).json({ error: err.message });
  }
};


module.exports = {
  register,
  login,
};