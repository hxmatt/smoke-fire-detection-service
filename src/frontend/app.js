// box colour per class
const colors = { fire: "#d85a30", smoke: "#5f5e5a" };

// canvas we draw the image and boxes onto
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// the held image and its file, kept for repeated runs
let image = null;
let file = null;

// fill the model dropdown from the API
fetch("/models")
  .then(r => r.json())
  .then(d => {
    document.getElementById("model").innerHTML = d.available_models
      .map(m => `<option value="${m.id}">${m.name}</option>`)
      .join("");
  });

// keep the readout in step with the slider
const conf = document.getElementById("conf");
conf.addEventListener("input", () => {
  document.getElementById("confVal").textContent = Number(conf.value).toFixed(2);
});

// drag-and-drop or click to choose an image
const upload = document.getElementById("upload");
const input = document.getElementById("file");

upload.addEventListener("click", () => input.click());

upload.addEventListener("dragover", e => {
  e.preventDefault();
  upload.classList.add("drag");
});

upload.addEventListener("dragleave", () => {
  upload.classList.remove("drag");
});

upload.addEventListener("drop", e => {
  e.preventDefault();
  upload.classList.remove("drag");
  if (e.dataTransfer.files[0]) load(e.dataTransfer.files[0]);
});

input.addEventListener("change", () => {
  if (input.files[0]) load(input.files[0]);
});

// load the chosen image once; it stays in the browser for repeated runs
function load(f) {
  file = f;
  upload.textContent = f.name;

  image = new Image();
  image.onload = () => {
    // size the canvas to the image, then swap it in for the placeholder
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    document.getElementById("view").hidden = true;
    canvas.hidden = false;

    ctx.drawImage(image, 0, 0);
    document.getElementById("run").disabled = false;
  };
  image.src = URL.createObjectURL(f);
}

// run detection on the held image with the current model and threshold
document.getElementById("run").addEventListener("click", async () => {
  const btn = document.getElementById("run");
  btn.disabled = true;
  btn.textContent = "Detecting...";

  const body = new FormData();
  body.append("image", file);
  body.append("model_id", document.getElementById("model").value);
  body.append("conf", conf.value);

  try {
    // send the image and get the detections back
    const res = await fetch("/predict", { method: "POST", body });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);

    const out = await res.json();
    render(out.detections);

    // update the stats panel
    document.getElementById("sTime").textContent = Math.round(out.inference_time_ms) + " ms";
    document.getElementById("sCount").textContent = out.detections.length;
  } catch (err) {
    alert(`Detection failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Detect";
  }
});

// redraw the image with a labelled box per detection, then list them
function render(dets) {
  ctx.drawImage(image, 0, 0);

  // scale line width and font to the image so boxes read at any resolution
  const lw = Math.max(2, canvas.width / 320);
  const fs = Math.max(13, canvas.width / 40);
  ctx.lineWidth = lw;
  ctx.font = `600 ${fs}px system-ui, sans-serif`;
  ctx.textBaseline = "bottom";

  for (const d of dets) {
    const [x, y, w, h] = d.bbox;
    const c = colors[d.label] || "#d94f1a";

    // the box
    ctx.strokeStyle = c;
    ctx.strokeRect(x, y, w, h);

    // the filled label tag above it
    const tag = `${d.label} ${d.confidence.toFixed(2)}`;
    const tw = ctx.measureText(tag).width;
    ctx.fillStyle = c;
    ctx.fillRect(x, y - fs - 4, tw + 8, fs + 4);
    ctx.fillStyle = "#fff";
    ctx.fillText(tag, x + 4, y - 1);
  }

  // list the detections below the canvas, one row with a confidence bar each
  const list = document.getElementById("list");
  list.innerHTML = dets.map(d => {
    const c = colors[d.label] || "#d94f1a";
    const pct = Math.round(d.confidence * 100);

    return `<div class="item">
      <span class="dot" style="background:${c}"></span>
      <span class="lbl">${d.label}</span>
      <span class="track"><div style="width:${pct}%;background:${c}"></div></span>
      <span class="pct">${d.confidence.toFixed(2)}</span>
    </div>`;
  }).join("");
}
