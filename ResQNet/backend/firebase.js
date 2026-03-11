const admin = require("firebase-admin");
const serviceAccount = require("./serviceAccountKey.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  projectId: "safenet-9e51e"
});

const db = admin.firestore();
module.exports = { db, admin };
