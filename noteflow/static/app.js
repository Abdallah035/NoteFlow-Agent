// Chat page. Sends messages to /chat and draws the answer.

const messagesBox = document.getElementById("messages");
const optionsBox = document.getElementById("options");
const form = document.getElementById("form");
const input = document.getElementById("input");
const notesBox = document.getElementById("notes");
const seedButton = document.getElementById("seed");

// Show one message bubble.
function addMessage(text, who) {
  const bubble = document.createElement("div");
  bubble.className = "msg " + who;

  const inner = document.createElement("div");
  inner.dir = "auto";              // makes Arabic read right to left
  inner.textContent = text;

  bubble.appendChild(inner);
  messagesBox.appendChild(bubble);
  messagesBox.scrollTop = messagesBox.scrollHeight;
}

// Draw the buttons for a choice or a yes/no question.
function showOptions(options) {
  optionsBox.innerHTML = "";

  options.forEach(function (option) {
    const button = document.createElement("button");
    button.textContent = option.label;

    // Colour the yes and no buttons.
    if (option.value === "yes") button.className = "yes";
    if (option.value === "no") button.className = "no";

    button.onclick = function () {
      optionsBox.innerHTML = "";
      send(option.value, option.label);
    };

    optionsBox.appendChild(button);
  });
}

// Send a message to the server.
async function send(message, labelToShow) {
  addMessage(labelToShow || message, "user");
  optionsBox.innerHTML = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message })
    });

    const answer = await response.json();
    addMessage(answer.reply, "agent");

    if (answer.options && answer.options.length > 0) {
      showOptions(answer.options);
    }

    loadNotes();
  } catch (error) {
    addMessage("Could not reach the server.", "agent");
  }
}

// Refresh the sidebar list.
async function loadNotes() {
  const response = await fetch("/notes");
  const data = await response.json();

  notesBox.innerHTML = "";

  if (data.notes.length === 0) {
    notesBox.innerHTML = '<div class="empty">No notes yet</div>';
    return;
  }

  data.notes.forEach(function (note) {
    const box = document.createElement("div");
    box.className = "note";

    const tags = note.tags
      .map(function (tag) { return '<span class="tag">' + tag + "</span>"; })
      .join("");

    box.innerHTML =
      '<div class="title" dir="auto"></div>' +
      '<div class="body" dir="auto"></div>' +
      '<div class="meta">' + tags + " #" + note.id + " - " +
      note.created_at.slice(0, 10) + "</div>";

    // textContent avoids HTML injection from note text.
    box.querySelector(".title").textContent = note.title;
    box.querySelector(".body").textContent = note.body.slice(0, 70);

    notesBox.appendChild(box);
  });
}

form.onsubmit = function (event) {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  send(message);
};

seedButton.onclick = async function () {
  await fetch("/seed", { method: "POST" });
  loadNotes();
  addMessage("Demo notes added. Try: delete the note about API", "agent");
};

loadNotes();
