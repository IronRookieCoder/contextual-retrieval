document.addEventListener("submit", (event) => {
  const form = event.target;
  if (!form.matches("[data-wait]")) {
    return;
  }

  form.classList.add("is-waiting");
  const button = form.querySelector("button[type='submit']");
  if (button) {
    button.dataset.originalText = button.textContent;
    button.textContent = "执行中";
    button.disabled = true;
  }
});
