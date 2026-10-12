(function () {
  "use strict";

  const state = {
    direction: null,
    pairs: [],
    live: false,
    pos: 50,
    dragging: false,
  };

  const dirGrid = document.getElementById("dirGrid");
  const sampleNote = document.getElementById("sampleNote");
  const compareWrap = document.getElementById("compareWrap");
  const emptyState = document.getElementById("emptyState");
  const gallery = document.getElementById("gallery");
  const galleryEmpty = document.getElementById("galleryEmpty");
  const beforeImg = document.getElementById("beforeImg");
  const afterImg = document.getElementById("afterImg");
  const beforePane = document.getElementById("beforePane");
  const sliderHandle = document.getElementById("sliderHandle");
  const compare = document.getElementById("compare");
  const tagBefore = document.getElementById("tagBefore");
  const tagAfter = document.getElementById("tagAfter");
  const compareCaption = document.getElementById("compareCaption");
  const inferForm = document.getElementById("inferForm");
  const imageInput = document.getElementById("imageInput");
  const fileName = document.getElementById("fileName");
  const inferBtn = document.getElementById("inferBtn");
  const inferMsg = document.getElementById("inferMsg");
  const dropzone = document.getElementById("dropzone");
  const uploadHint = document.getElementById("uploadHint");
  const modeBadge = document.getElementById("modeBadge");

  function showMsg(text, kind) {
    inferMsg.hidden = false;
    inferMsg.className = "msg " + (kind || "info");
    inferMsg.textContent = text;
  }

  function setSlider(pos) {
    state.pos = Math.max(0, Math.min(100, pos));
    beforePane.style.width = state.pos + "%";
    sliderHandle.style.left = state.pos + "%";
    sliderHandle.setAttribute("aria-valuenow", String(Math.round(state.pos)));
    syncBeforeImageWidth();
  }

  function syncBeforeImageWidth() {
    const w = compare.clientWidth;
    beforeImg.style.width = w + "px";
    beforeImg.style.height = compare.clientHeight + "px";
  }

  function pointerPos(evt) {
    const rect = compare.getBoundingClientRect();
    const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
    return ((clientX - rect.left) / rect.width) * 100;
  }

  function startDrag(evt) {
    state.dragging = true;
    setSlider(pointerPos(evt));
    evt.preventDefault();
  }
  function moveDrag(evt) {
    if (!state.dragging) return;
    setSlider(pointerPos(evt));
  }
  function endDrag() {
    state.dragging = false;
  }

  compare.addEventListener("mousedown", startDrag);
  window.addEventListener("mousemove", moveDrag);
  window.addEventListener("mouseup", endDrag);
  compare.addEventListener("touchstart", startDrag, { passive: false });
  window.addEventListener("touchmove", moveDrag, { passive: false });
  window.addEventListener("touchend", endDrag);
  window.addEventListener("resize", syncBeforeImageWidth);

  sliderHandle.addEventListener("keydown", function (e) {
    if (e.key === "ArrowLeft") setSlider(state.pos - 3);
    if (e.key === "ArrowRight") setSlider(state.pos + 3);
  });

  function showCompare(pair) {
    emptyState.hidden = true;
    compareWrap.hidden = false;
    beforeImg.src = pair.before_url;
    afterImg.src = pair.after_url;
    tagBefore.textContent = pair.before_label || "输入";
    tagAfter.textContent = pair.after_label || "输出";
    compareCaption.textContent = pair.caption || "";
    setSlider(50);
    beforeImg.onload = syncBeforeImageWidth;
  }

  function renderGallery(pairs) {
    gallery.innerHTML = "";
    if (!pairs.length) {
      galleryEmpty.hidden = false;
      return;
    }
    galleryEmpty.hidden = true;
    pairs.forEach(function (p, idx) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "thumb" + (idx === 0 ? " active" : "");
      btn.innerHTML =
        '<img src="' +
        p.before_url +
        '" alt="' +
        p.title +
        '" /><span>' +
        p.title +
        "</span>";
      btn.addEventListener("click", function () {
        Array.prototype.forEach.call(gallery.querySelectorAll(".thumb"), function (el) {
          el.classList.remove("active");
        });
        btn.classList.add("active");
        showCompare(p);
      });
      gallery.appendChild(btn);
    });
  }

  function updateUploadUI(live) {
    inferBtn.disabled = !(live && imageInput.files && imageInput.files[0]);
    if (live) {
      uploadHint.textContent =
        "已检测到该方向权重：上传图片后点击「运行推理」。推理结果将显示在对比区域。";
    } else {
      uploadHint.textContent =
        "当前方向无权重，在线推理不可用。请使用下方展示样例；或将 latest_net_G.pth 放到对应 checkpoints 目录后重启服务。";
    }
  }

  async function selectDirection(id) {
    state.direction = id;
    Array.prototype.forEach.call(dirGrid.querySelectorAll(".dir-card"), function (card) {
      card.setAttribute("aria-pressed", card.dataset.direction === id ? "true" : "false");
    });

    sampleNote.textContent = "加载中…";
    gallery.innerHTML = "";
    compareWrap.hidden = true;
    emptyState.hidden = false;

    try {
      const res = await fetch("/api/samples/" + encodeURIComponent(id));
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "加载失败");
      state.pairs = data.pairs || [];
      state.live = !!data.live;
      sampleNote.textContent = data.note;
      updateUploadUI(state.live);
      renderGallery(state.pairs);
      if (state.pairs.length) {
        showCompare(state.pairs[0]);
      } else {
        emptyState.textContent = data.note || "暂无样例";
        emptyState.hidden = false;
      }
    } catch (err) {
      sampleNote.textContent = String(err.message || err);
      showMsg(sampleNote.textContent, "error");
    }
  }

  dirGrid.addEventListener("click", function (e) {
    const card = e.target.closest(".dir-card");
    if (!card) return;
    selectDirection(card.dataset.direction);
  });

  imageInput.addEventListener("change", function () {
    const f = imageInput.files && imageInput.files[0];
    fileName.textContent = f ? f.name : "";
    updateUploadUI(state.live);
  });

  ["dragenter", "dragover"].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", function (e) {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files[0]) {
      imageInput.files = files;
      fileName.textContent = files[0].name;
      updateUploadUI(state.live);
    }
  });

  inferForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    if (!state.direction) {
      showMsg("请先选择迁移方向", "error");
      return;
    }
    if (!imageInput.files || !imageInput.files[0]) {
      showMsg("请先选择图片", "error");
      return;
    }
    if (!state.live) {
      showMsg(
        "未找到权重，无法在线推理。请使用展示模式，或放置 .pth 后重启服务。",
        "info"
      );
      return;
    }

    const fd = new FormData();
    fd.append("direction", state.direction);
    fd.append("image", imageInput.files[0]);
    inferBtn.disabled = true;
    showMsg("推理中，请稍候…", "info");

    try {
      const res = await fetch("/api/infer", { method: "POST", body: fd });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "推理失败");
      showMsg("推理完成（在线模式）", "ok");
      showCompare({
        before_url: data.before_url,
        after_url: data.after_url,
        before_label: data.before_label,
        after_label: data.after_label,
        caption: "在线推理结果 · " + (state.direction || ""),
      });
    } catch (err) {
      showMsg(String(err.message || err), "error");
    } finally {
      updateUploadUI(state.live);
    }
  });

  // 默认选中第一个有展示样例的方向
  const preferred =
    dirGrid.querySelector('.dir-card[data-direction="sar2opt"]') ||
    dirGrid.querySelector(".dir-card");
  if (preferred) selectDirection(preferred.dataset.direction);

  // 刷新状态角标（可选）
  fetch("/api/status")
    .then(function (r) {
      return r.json();
    })
    .then(function (data) {
      if (!data.ok) return;
      modeBadge.textContent = data.any_live ? "在线推理可用" : "展示模式（无权重）";
      modeBadge.classList.toggle("live", !!data.any_live);
      modeBadge.classList.toggle("showcase", !data.any_live);
    })
    .catch(function () {});
})();
