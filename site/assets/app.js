/* Shared page behaviour: theme toggle, copy-to-clipboard, footer year.
   Loaded with `defer` from every page; no external dependencies. */

const STORAGE_KEY = "abi-theme";

function systemTheme() {
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function currentTheme() {
  return document.documentElement.dataset.theme || systemTheme();
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* private mode or blocked storage: the toggle still works for this page view */
  }
  const toggle = document.querySelector("[data-theme-toggle]");
  if (toggle) toggle.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} theme`);
}

function initTheme() {
  const toggle = document.querySelector("[data-theme-toggle]");
  if (!toggle) return;
  applyTheme(currentTheme());
  toggle.addEventListener("click", () => applyTheme(currentTheme() === "dark" ? "light" : "dark"));
}

function initCopyButtons() {
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const block = button.closest(".code")?.querySelector("pre");
      if (!block) return;
      const text = block.innerText.replace(/^\$ /gm, "");
      const original = button.textContent;
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Copy failed";
      }
      setTimeout(() => {
        button.textContent = original;
      }, 1600);
    });
  });
}

function initYear() {
  const slot = document.querySelector("[data-year]");
  if (slot) slot.textContent = String(new Date().getFullYear());
}

initTheme();
initCopyButtons();
initYear();
