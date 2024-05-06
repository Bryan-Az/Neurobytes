require("dotenv").config({ path: "../.env" });
const { Sequelize, DataTypes } = require("sequelize");

const sequelize = new Sequelize({
  dialect: "mysql",
  host: process.env.DB_HOST,
  username: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
});

const models = {
  // User: require("./models/user.model")(sequelize, DataTypes),
};

Object.values(models)
  .filter((model) => typeof model.associate === "function")
  .forEach((model) => model.associate(models));

const initializeDatabase = async () => {
  try {
    await sequelize.sync();
    console.log("Database & tables created!");
    return { User: models.User };
  } catch (error) {
    console.error("Error initializing database:", error);
    throw error;
  }
};

module.exports = {
  initializeDatabase,
  ...models,
};
