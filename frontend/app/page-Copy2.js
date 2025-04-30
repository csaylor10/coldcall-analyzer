"use client";
import { useState } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [downloads, setDownloads] = useState([]);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("idle"); // 'idle' | 'uploading' | 'analyzing'

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length > 1000) {
      setError("❌ You can upload up to 1000 files at a time.");
      return;
    }

    setError("");
    setDownloads([]);
    setLoading(true);
    setProgress(0);
    setStatus("uploading"); // Start with uploading clearly

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
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              setProgress(percentCompleted);
            },
          }
        );

        setStatus("analyzing"); // Clearly update status to analyzing immediately after upload finishes

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

        const blob = new Blob([`
          <!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Analysis Report: ${filename}</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background-color: #f7f9fc;
      color: #333;
      padding: 40px;
      margin: 0;
    }

    .container {
      max-width: 850px;
      margin: auto;
      background-color: #ffffff;
      border-radius: 10px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      padding: 25px;
      overflow: hidden;
    }

    h1 {
      color: #0056b3;
      margin-bottom: 20px;
      border-bottom: 2px solid #eaeaea;
      padding-bottom: 10px;
    }

    pre {
      white-space: pre-wrap;
      word-wrap: break-word;
      background-color: #fafafa;
      border-radius: 5px;
      padding: 15px;
      border-left: 5px solid #0056b3;
      font-size: 15px;
      line-height: 1.5;
      color: #444;
    }

    .footer {
      text-align: center;
      font-size: 12px;
      color: #aaa;
      margin-top: 30px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>📊 Analysis Report: ${filename}</h1>
    <pre>${full_analysis_content}</pre>
    <div class="footer">Generated automatically by Greener Living AI Analysis.</div>
  </div>
</body>
</html>
        `], { type: "text/html" });

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

      } catch (err) {
  console.error("Detailed error:", err.response || err.message || err);
  setError(`❌ Error processing file ${file.name}: ${err.response?.data?.detail || err.message}`);
        }

    }
    setLoading(false);
    setStatus("idle");
    setProgress(0);
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    multiple: true,
    maxFiles: 1000,
  });

  return (
    <div className="p-10 bg-white text-black min-h-screen">
      <div {...getRootProps()} className="border-2 border-dashed p-6 rounded cursor-pointer text-center">
        <input {...getInputProps()} multiple />
        <p>Drag & drop audio files (up to 1000), or click to select files.</p>
      </div>

      {status === "uploading" && (
        <>
          <div className="mt-4 w-full bg-gray-200 rounded-full h-4">
            <div
              className="bg-green-500 h-4 rounded-full transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="mt-2">⏳ Upload Progress: {progress}%</p>
        </>
      )}

      {status === "analyzing" && (
        <p className="mt-4">🤖 Analyzing... (This may take a moment.)</p>
      )}

      {error && <p className="mt-4 text-red-500">{error}</p>}

      {downloads.length > 0 && (
        <div className="mt-4 p-4 bg-gray-100 rounded shadow">
          <h2 className="font-semibold">📁 Download Analysis Reports:</h2>
          <ul>
            {downloads.map((file, idx) => (
              <li key={idx}>
                <a href={file.url} download={file.name} className="text-blue-600 underline">
                  {file.name}
                </a>
              </li>
            ))}
          </ul>

          <h2 className="mt-4 font-semibold text-red-600">🚩 Red Flags Detected:</h2>
          <ul>
            {downloads.map((file, idx) => (
              <li key={`flag-${idx}`} className="mb-2">
                <strong>{file.name}</strong>: {file.red_flags}<br />
                <em>Reason:</em> {file.red_flag_reason || "N/A"}<br />
                <em>Quotes:</em> {file.red_flag_quotes || "None"}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
