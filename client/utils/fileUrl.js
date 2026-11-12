const FILES_BASE = (
  import.meta.env.VITE_FILES_BASE_URL || "https://workingcart.com/files"
).replace(/\/+$/, "");

export const buildFileUrl = (p) => {
  if (!p || typeof p !== "string") return "";
  if (/^https?:\/\//i.test(p)) return p;
  const fileName = p.split(/[\\/]/).pop(); // sirf filename
  return `${FILES_BASE}/${fileName}`;
};