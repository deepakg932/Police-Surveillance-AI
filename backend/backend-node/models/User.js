const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
  fullName: {
    type: String,
  
   
  },

  email: {
    type: String,
  
    unique: true,
  },

  password: {
    type: String,
  
  },

  badgeNumber: {
    type: String,
   
    unique: true,
  },

  department: {
    type: String,
 
    enum: [
      "patrol",
      "detective",
      "traffic",
      "cyber",
      "special",
      "administration",
    ],
  },

  phone: {
    type: String,
    default: "",
  },

  role: {
    type: String,
    default: "officer",
  },

  createdAt: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model("User", userSchema);