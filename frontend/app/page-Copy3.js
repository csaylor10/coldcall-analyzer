"use client";
import { useState } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";

export default function Home() {
  const [error, setError] = useState("");
  const [downloads, setDownloads] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [completedAnalyses, setCompletedAnalyses] = useState(0);
  const [totalFiles, setTotalFiles] = useState(0);
  const [status, setStatus] = useState("idle");

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length > 1000) {
      setError("❌ You can upload up to 1000 files at a time.");
      return;
    }

    setError("");
    setDownloads([]);
    setUploadProgress(0);
    setCompletedAnalyses(0);
    setTotalFiles(acceptedFiles.length);
    setStatus("uploading");

    for (let file of acceptedFiles) {
      const formData = new FormData();
      formData.append("file", file);

      try {
        setStatus("uploading");

        const res = await axios.post(
          "https://pmcdnsk3jztr4h-8000.proxy.runpod.net/upload/",
          formData,
          {
            headers: { "Content-Type": "multipart/form-data" },
            onUploadProgress: (progressEvent) => {
              setUploadProgress(
                Math.round((progressEvent.loaded * 100) / progressEvent.total)
              );
            },
          }
        );

        setStatus("analyzing");

        const {
          full_analysis_content,
          overall_perf,
          rep_name,
          categorization,
          red_flags,
          red_flag_reason,
          red_flag_quotes,
        } = res.data;

        const filename = `${overall_perf} ${rep_name} ${categorization} ${file.name}.html`;

        const blob = new Blob(
          [
            `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Analysis Report: ${filename}</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; background-color: #f9fafb; color: #111827; padding: 20px; }
    .container { max-width: 900px; margin: auto; padding: 25px; background-color: #ffffff; box-shadow: 0 4px 10px rgba(0,0,0,0.1); border-radius: 8px; }
    h1, h2 { color: #0369a1; }
    .analysis-content { line-height: 1.6; font-size: 16px; margin-bottom: 20px; }
    .red-flags { background-color: #fef2f2; border-left: 5px solid #dc2626; padding: 15px; border-radius: 5px; margin-top: 20px; }
    blockquote { font-style: italic; background-color: #f3f4f6; padding: 10px; border-radius: 5px; }
    footer { margin-top: 30px; text-align: center; color: #6b7280; font-size: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>📊 Analysis Report: ${filename}</h1>
    <div class="analysis-content">
      ${full_analysis_content.replace(/\n/g, "<br />")}
    </div>

    ${
      red_flags === "Yes"
        ? `<div class="red-flags">
      <h2>🚩 Red Flags Detected</h2>
      <p><strong>Reason:</strong> ${
        red_flag_reason || "N/A"
      }</p>
      ${
        red_flag_quotes !== "None"
          ? `<blockquote>${red_flag_quotes}</blockquote>`
          : ""
      }
    </div>`
        : ""
    }

    <footer>Generated automatically by Greener Living AI Analysis.</footer>
  </div>
</body>
</html>`,
          ],
          { type: "text/html" }
        );

        const url = URL.createObjectURL(blob);

        setDownloads((prev) => [
          ...prev,
          {
            name: filename,
            url,
            red_flags,
            red_flag_reason,
            red_flag_quotes,
          },
        ]);

        setCompletedAnalyses((prev) => prev + 1);
      } catch (err) {
        setError(
          `❌ Error processing ${file.name}: ${
            err.response?.data?.detail || err.message
          }`
        );
        setCompletedAnalyses((prev) => prev + 1);
      }
    }

    setStatus("idle");
    setUploadProgress(0);
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    multiple: true,
    maxFiles: 1000,
  });

  return (
    <div className="p-10 bg-white text-black min-h-screen">
      <div
        {...getRootProps()}
        className="border-2 border-dashed p-6 rounded cursor-pointer text-center"
      >
        <input {...getInputProps()} />
        <p>
          Drag & drop audio files (up to 1000), or click to select files.
        </p>
      </div>

      {status === "uploading" && (
        <div className="mt-4">
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div
              className="bg-green-500 h-4 rounded-full transition-all"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <p className="mt-2">⏳ Upload Progress: {uploadProgress}%</p>
        </div>
      )}

      {(status === "analyzing" || completedAnalyses > 0) && (
        <div className="mt-4">
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div
              className="bg-blue-500 h-4 rounded-full transition-all"
              style={{ width: `${(completedAnalyses / totalFiles) * 100}%` }}
            />
          </div>
          <p className="mt-2">
            🔎 Analyses Completed: {completedAnalyses}/{totalFiles}
          </p>
        </div>
      )}

      {error && <p className="mt-4 text-red-500">{error}</p>}

      {downloads.length > 0 && (
        <div className="mt-4 p-4 bg-gray-100 rounded shadow">
          <h2 className="font-semibold">📁 Download Analysis Reports:</h2>
          <ul>
            {downloads.map((file, idx) => (
              <li key={idx} className="mb-4">
                <a
                  href={file.url}
                  download={file.name}
                  className="text-blue-600 underline"
                >
                  {file.name}
                </a>
                <div className="ml-4 mt-1">
                  <strong>🚩 Red Flags Detected:</strong> {file.red_flags}
                  <br />
                  <em>Reason:</em> {file.red_flag_reason || "None"}
                  <br />
                  <em>Quotes:</em> {file.red_flag_quotes || "None"}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
