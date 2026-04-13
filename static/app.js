const youtubeUrlInput = document.getElementById("youtubeUrl");
const transcribeButton = document.getElementById("transcribeButton");
const transcribeAndSummarizeButton = document.getElementById("transcribeAndSummarizeButton");
const summarizeButton = document.getElementById("summarizeButton");
const statusBanner = document.getElementById("statusBanner");
const transcriptText = document.getElementById("transcriptText");
const summaryText = document.getElementById("summaryText");
const videoTitle = document.getElementById("videoTitle");
const videoChannel = document.getElementById("videoChannel");
const transcriptMeta = document.getElementById("transcriptMeta");
const summaryMeta = document.getElementById("summaryMeta");

function setStatus(message, state) {
  statusBanner.textContent = message;
  statusBanner.className = `status-banner ${state}`;
}

function setButtonsDisabled(isDisabled) {
  transcribeButton.disabled = isDisabled;
  transcribeAndSummarizeButton.disabled = isDisabled;
  summarizeButton.disabled = isDisabled || !transcriptText.value.trim();
}

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) {
    return "Unknown duration";
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
}

function readErrorMessage(payload, fallbackMessage) {
  if (payload && typeof payload.detail === "string") {
    return payload.detail;
  }
  return fallbackMessage;
}

async function postJson(url, payload, fallbackMessage) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(readErrorMessage(data, fallbackMessage));
  }

  return data;
}

function renderTranscript(result) {
  const source = result.source || {};
  transcriptText.value = result.full_text || "";
  summaryText.textContent = "No summary yet. Generate it when you are ready.";
  summaryText.classList.remove("muted");
  summarizeButton.disabled = !transcriptText.value.trim();

  videoTitle.textContent = source.title || "Untitled video";
  videoTitle.classList.remove("muted");

  const channel = source.channel || "Unknown channel";
  const duration = formatDuration(result.duration || source.duration);
  videoChannel.textContent = `${channel} • ${duration}`;
  videoChannel.classList.remove("muted");

  transcriptMeta.textContent = `${result.word_count || 0} words • ${result.language || "unknown language"}`;
  transcriptMeta.classList.remove("muted");

  summaryMeta.textContent = "Not generated";
  summaryMeta.classList.add("muted");
}

function renderSummary(summary) {
  summaryText.textContent = summary.summary || "Summary generation returned no text.";
  summaryText.classList.remove("muted");

  const parts = [];
  if (summary.chunk_count) {
    parts.push(`${summary.chunk_count} chunk${summary.chunk_count === 1 ? "" : "s"}`);
  }
  if (summary.model) {
    parts.push(summary.model);
  }

  summaryMeta.textContent = parts.join(" • ") || "Summary ready";
  summaryMeta.classList.remove("muted");
}

async function transcribeOnly() {
  const url = youtubeUrlInput.value.trim();
  if (!url) {
    setStatus("Paste a YouTube URL first.", "error");
    return;
  }

  setButtonsDisabled(true);
  setStatus("Downloading audio and running Whisper transcription. This can take a while on longer videos.", "loading");

  try {
    const result = await postJson(
      "/transcribe-url",
      { url },
      "Transcription failed.",
    );

    renderTranscript(result);
    setStatus("Transcript ready. You can review it or generate a summary now.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonsDisabled(false);
  }
}

async function transcribeAndSummarize() {
  const url = youtubeUrlInput.value.trim();
  if (!url) {
    setStatus("Paste a YouTube URL first.", "error");
    return;
  }

  setButtonsDisabled(true);
  setStatus("Downloading, transcribing, and summarizing the video. Longer videos will take more time.", "loading");

  try {
    const result = await postJson(
      "/transcribe-url-and-summarize",
      { url },
      "Processing failed.",
    );

    renderTranscript(result);
    renderSummary(result.summary || {});
    setStatus("Transcript and summary are ready.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonsDisabled(false);
  }
}

async function summarizeExistingTranscript() {
  const text = transcriptText.value.trim();
  if (!text) {
    setStatus("There is no transcript to summarize yet.", "error");
    summarizeButton.disabled = true;
    return;
  }

  setButtonsDisabled(true);
  setStatus("Summarizing the transcript in chunks for a cleaner overview.", "loading");

  try {
    const summary = await postJson(
      "/summarize",
      { text },
      "Summary generation failed.",
    );

    renderSummary(summary);
    setStatus("Summary ready.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setButtonsDisabled(false);
  }
}

transcribeButton.addEventListener("click", transcribeOnly);
transcribeAndSummarizeButton.addEventListener("click", transcribeAndSummarize);
summarizeButton.addEventListener("click", summarizeExistingTranscript);
transcriptText.addEventListener("input", () => {
  summarizeButton.disabled = !transcriptText.value.trim();
});
