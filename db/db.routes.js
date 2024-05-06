const express = require("express");
const router = express.Router();
const dbController = require("./db.controller");

module.exports = router;

router.get("/example", dbController.example); // placeholder for actual route