const Sequelize = require("sequelize");

const {} = require("../db/db");

const example = async (req, res) => {
  const { ex } = req.body;
  try {
    // const example = await Example.findAll();
    res.status(200).json({ message: "Example route", ex });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

module.exports = {};
