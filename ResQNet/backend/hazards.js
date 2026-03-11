const { db, admin } = require("./firebase");

async function addHazard(type, lat, lng, meta = {}) {
  try {
    const doc = await db.collection("hazards").add({
      type:      type,
      location:  new admin.firestore.GeoPoint(lat, lng),
      timestamp: admin.firestore.FieldValue.serverTimestamp(),
      ...meta
    });
    console.log(`✅ Firestore [safenet-9e51e]: ${type} written → ID: ${doc.id}`);
    return doc.id;
  } catch (err) {
    console.error(`❌ Firestore write failed:`, err.message);
    return null;
  }
}

module.exports = { addHazard };







