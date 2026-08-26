const path = require("path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Silences a Turbopack warning: it walks up from this dir looking for a workspace
  // root and finds an unrelated package-lock.json outside this git repo (a sibling
  // dev artifact on this machine, nothing to do with this project). Pinning the root
  // here to this directory is the fix Next.js itself suggests for that warning.
  turbopack: {
    root: path.join(__dirname),
  },
};

module.exports = nextConfig;
