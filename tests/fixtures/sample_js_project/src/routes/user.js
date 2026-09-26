class UserController {
  constructor(db) {
    this.db = db;
  }

  async getUserById(userId) {
    return await this.db.find(userId);
  }
}

function createRouter(controller) {
  return { controller };
}

module.exports = { UserController, createRouter };
