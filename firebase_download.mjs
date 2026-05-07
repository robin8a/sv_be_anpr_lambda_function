import { initializeApp } from "firebase/app";
import { getStorage, ref, getBytes } from "firebase/storage";

function requiredEnv(name) {
  const v = process.env[name];
  if (!v) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return v;
}

async function main() {
  const imageFirebaseKey = process.argv[2];
  if (!imageFirebaseKey) {
    console.error("Usage: node firebase_download.mjs <image_firebase_key>");
    process.exit(2);
  }

  const firebaseConfig = {
    apiKey: requiredEnv("sv_apiKey"),
    authDomain: requiredEnv("sv_authDomain"),
    databaseURL: requiredEnv("sv_databaseURL"),
    projectId: requiredEnv("sv_projectId"),
    storageBucket: requiredEnv("sv_storageBucket"),
    messagingSenderId: requiredEnv("sv_messagingSenderId"),
    appId: requiredEnv("sv_appId"),
    measurementId: process.env.sv_measurementId || undefined,
  };

  const app = initializeApp(firebaseConfig);
  const storage = getStorage(app);
  const fileRef = ref(storage, imageFirebaseKey);

  // Firebase Storage client SDK `getBytes` returns a Uint8Array.
  // We print base64 so the Python handler can decode it safely.
  const bytes = await getBytes(fileRef);
  const b64 = Buffer.from(bytes).toString("base64");
  process.stdout.write(b64);
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(1);
});

