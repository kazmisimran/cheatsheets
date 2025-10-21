# Front-End Web Development Cheatsheet

## HTML (HyperText Markup Language)

### Basic Structure
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Title</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <h1>Hello World</h1>
    <script src="script.js"></script>
</body>
</html>
```

### Text Elements
```html
<h1>Heading 1</h1>  <!-- h1 to h6 -->
<p>Paragraph text</p>
<strong>Bold text</strong>
<em>Italic text</em>
<br>  <!-- Line break -->
<hr>  <!-- Horizontal rule -->
<pre>Preformatted text</pre>
<code>Code snippet</code>
<blockquote>Quote</blockquote>
```

### Links and Images
```html
<a href="https://example.com">Link text</a>
<a href="#section">Internal link</a>
<a href="mailto:email@example.com">Email link</a>
<a href="tel:+1234567890">Phone link</a>

<img src="image.jpg" alt="Description" width="300" height="200">
```

### Lists
```html
<!-- Unordered list -->
<ul>
    <li>Item 1</li>
    <li>Item 2</li>
</ul>

<!-- Ordered list -->
<ol>
    <li>First item</li>
    <li>Second item</li>
</ol>

<!-- Description list -->
<dl>
    <dt>Term</dt>
    <dd>Definition</dd>
</dl>
```

### Tables
```html
<table>
    <thead>
        <tr>
            <th>Header 1</th>
            <th>Header 2</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Data 1</td>
            <td>Data 2</td>
        </tr>
    </tbody>
</table>
```

### Forms
```html
<form action="/submit" method="POST">
    <label for="name">Name:</label>
    <input type="text" id="name" name="name" required>
    
    <label for="email">Email:</label>
    <input type="email" id="email" name="email" required>
    
    <label for="password">Password:</label>
    <input type="password" id="password" name="password">
    
    <label for="age">Age:</label>
    <input type="number" id="age" min="0" max="120">
    
    <label for="date">Date:</label>
    <input type="date" id="date">
    
    <label for="color">Color:</label>
    <input type="color" id="color">
    
    <label for="file">File:</label>
    <input type="file" id="file">
    
    <label for="message">Message:</label>
    <textarea id="message" rows="4"></textarea>
    
    <label for="country">Country:</label>
    <select id="country">
        <option value="us">United States</option>
        <option value="uk">United Kingdom</option>
    </select>
    
    <label>
        <input type="checkbox" name="subscribe"> Subscribe
    </label>
    
    <label>
        <input type="radio" name="gender" value="male"> Male
    </label>
    <label>
        <input type="radio" name="gender" value="female"> Female
    </label>
    
    <button type="submit">Submit</button>
</form>
```

### Semantic HTML5 Elements
```html
<header>Header content</header>
<nav>Navigation links</nav>
<main>Main content</main>
<section>Section of content</section>
<article>Article content</article>
<aside>Sidebar content</aside>
<footer>Footer content</footer>
<figure>
    <img src="image.jpg" alt="Description">
    <figcaption>Image caption</figcaption>
</figure>
```

### Media Elements
```html
<video controls width="600">
    <source src="video.mp4" type="video/mp4">
    Your browser doesn't support video.
</video>

<audio controls>
    <source src="audio.mp3" type="audio/mpeg">
    Your browser doesn't support audio.
</audio>

<iframe src="https://example.com" width="600" height="400"></iframe>
```

---

## CSS (Cascading Style Sheets)

### Selectors
```css
/* Element selector */
p { color: blue; }

/* Class selector */
.class-name { color: red; }

/* ID selector */
#id-name { color: green; }

/* Universal selector */
* { margin: 0; }

/* Descendant selector */
div p { color: blue; }

/* Child selector */
div > p { color: red; }

/* Adjacent sibling */
h1 + p { margin-top: 0; }

/* Attribute selector */
input[type="text"] { border: 1px solid black; }

/* Pseudo-class */
a:hover { color: red; }
li:first-child { font-weight: bold; }
li:last-child { border: none; }
li:nth-child(odd) { background: #f0f0f0; }

/* Pseudo-element */
p::before { content: "→ "; }
p::after { content: " ←"; }
p::first-letter { font-size: 2em; }
```

### Box Model
```css
.box {
    width: 300px;
    height: 200px;
    padding: 20px;        /* Inside spacing */
    border: 2px solid black;
    margin: 10px;         /* Outside spacing */
    box-sizing: border-box; /* Include padding/border in width */
}
```

### Colors
```css
.element {
    color: red;                    /* Name */
    color: #ff0000;                /* Hex */
    color: rgb(255, 0, 0);         /* RGB */
    color: rgba(255, 0, 0, 0.5);   /* RGBA with transparency */
    color: hsl(0, 100%, 50%);      /* HSL */
    color: hsla(0, 100%, 50%, 0.5); /* HSLA */
}
```

### Typography
```css
.text {
    font-family: Arial, sans-serif;
    font-size: 16px;
    font-weight: bold;     /* 100-900 or bold/normal */
    font-style: italic;
    line-height: 1.5;
    text-align: center;    /* left, right, center, justify */
    text-decoration: underline;
    text-transform: uppercase; /* lowercase, capitalize */
    letter-spacing: 2px;
    word-spacing: 5px;
}
```

### Backgrounds
```css
.element {
    background-color: #f0f0f0;
    background-image: url('image.jpg');
    background-size: cover;        /* contain, 100px, 50% */
    background-position: center;
    background-repeat: no-repeat;  /* repeat, repeat-x, repeat-y */
    background-attachment: fixed;  /* scroll, fixed */
    
    /* Shorthand */
    background: #f0f0f0 url('image.jpg') no-repeat center/cover;
}
```

### Borders
```css
.element {
    border: 2px solid black;
    border-radius: 10px;           /* Rounded corners */
    border-top: 1px dashed red;
    border-right: 2px solid blue;
    border-bottom: 3px dotted green;
    border-left: 4px double orange;
}
```

### Flexbox
```css
.container {
    display: flex;
    flex-direction: row;           /* row, column, row-reverse, column-reverse */
    justify-content: center;       /* flex-start, flex-end, space-between, space-around */
    align-items: center;           /* flex-start, flex-end, stretch, baseline */
    flex-wrap: wrap;               /* nowrap, wrap, wrap-reverse */
    gap: 10px;                     /* Space between items */
}

.item {
    flex: 1;                       /* flex-grow, flex-shrink, flex-basis */
    flex-grow: 1;
    flex-shrink: 0;
    flex-basis: 200px;
    align-self: flex-start;
}
```

### Grid
```css
.container {
    display: grid;
    grid-template-columns: 1fr 2fr 1fr;    /* Fraction units */
    grid-template-columns: repeat(3, 1fr); /* Repeat */
    grid-template-rows: 100px auto 100px;
    gap: 20px;                             /* row-gap, column-gap */
    grid-template-areas:
        "header header header"
        "sidebar main main"
        "footer footer footer";
}

.item {
    grid-column: 1 / 3;            /* Start / End */
    grid-row: 1 / 2;
    grid-area: header;
}
```

### Positioning
```css
.element {
    position: static;              /* Default */
    position: relative;            /* Relative to normal position */
    position: absolute;            /* Relative to nearest positioned ancestor */
    position: fixed;               /* Relative to viewport */
    position: sticky;              /* Hybrid relative/fixed */
    
    top: 10px;
    right: 20px;
    bottom: 30px;
    left: 40px;
    z-index: 100;                  /* Stacking order */
}
```

### Display
```css
.element {
    display: block;                /* Takes full width */
    display: inline;               /* Takes only necessary width */
    display: inline-block;         /* Inline but with width/height */
    display: none;                 /* Hidden */
    display: flex;
    display: grid;
}
```

### Transitions and Animations
```css
/* Transition */
.button {
    background: blue;
    transition: background 0.3s ease;
}
.button:hover {
    background: red;
}

/* Animation */
@keyframes slide {
    from { transform: translateX(0); }
    to { transform: translateX(100px); }
}

.element {
    animation: slide 2s ease-in-out infinite;
}
```

### Transforms
```css
.element {
    transform: translate(50px, 100px);
    transform: rotate(45deg);
    transform: scale(1.5);
    transform: skew(10deg, 20deg);
    transform: translateX(100px) rotate(45deg) scale(1.2);
}
```

### Media Queries (Responsive Design)
```css
/* Mobile first */
.container {
    width: 100%;
}

/* Tablet */
@media (min-width: 768px) {
    .container {
        width: 750px;
    }
}

/* Desktop */
@media (min-width: 1024px) {
    .container {
        width: 960px;
    }
}

/* Print */
@media print {
    .no-print {
        display: none;
    }
}
```

### CSS Variables
```css
:root {
    --primary-color: #3498db;
    --secondary-color: #2ecc71;
    --font-size: 16px;
}

.element {
    color: var(--primary-color);
    font-size: var(--font-size);
}
```

---

## JavaScript

### Variables
```javascript
// var - function scoped (avoid)
var x = 10;

// let - block scoped, can be reassigned
let y = 20;
y = 30;

// const - block scoped, cannot be reassigned
const z = 40;
```

### Data Types
```javascript
// Primitives
let str = "Hello";              // String
let num = 42;                   // Number
let bool = true;                // Boolean
let nothing = null;             // Null
let undef;                      // Undefined
let sym = Symbol("id");         // Symbol
let bigNum = 123n;              // BigInt

// Objects
let obj = { name: "John", age: 30 };
let arr = [1, 2, 3, 4, 5];
let func = function() { return "Hello"; };
```

### Operators
```javascript
// Arithmetic
+ - * / % **                    // Addition, subtraction, multiplication, division, modulo, exponentiation

// Comparison
== === != !== > < >= <=         // Equal, strict equal, not equal, strict not equal

// Logical
&& || !                         // AND, OR, NOT

// Assignment
= += -= *= /= %=                // Assign, add assign, subtract assign, etc.

// Ternary
let result = (age >= 18) ? "Adult" : "Minor";

// Nullish coalescing
let name = username ?? "Guest"; // Use "Guest" if username is null/undefined

// Optional chaining
let street = user?.address?.street; // Safe navigation
```

### Strings
```javascript
let str = "Hello World";

str.length;                     // 11
str[0];                         // "H"
str.charAt(0);                  // "H"
str.toUpperCase();              // "HELLO WORLD"
str.toLowerCase();              // "hello world"
str.trim();                     // Remove whitespace
str.split(" ");                 // ["Hello", "World"]
str.slice(0, 5);                // "Hello"
str.substring(0, 5);            // "Hello"
str.replace("World", "JS");     // "Hello JS"
str.includes("World");          // true
str.startsWith("Hello");        // true
str.endsWith("World");          // true
str.indexOf("o");               // 4
str.repeat(3);                  // "Hello WorldHello WorldHello World"

// Template literals
let name = "John";
let greeting = `Hello, ${name}!`; // "Hello, John!"
```

### Arrays
```javascript
let arr = [1, 2, 3, 4, 5];

// Basic methods
arr.length;                     // 5
arr.push(6);                    // Add to end
arr.pop();                      // Remove from end
arr.unshift(0);                 // Add to beginning
arr.shift();                    // Remove from beginning
arr.splice(1, 2);               // Remove 2 items at index 1
arr.slice(1, 3);                // Return copy from index 1 to 3

// Iteration
arr.forEach(item => console.log(item));
arr.map(x => x * 2);            // [2, 4, 6, 8, 10]
arr.filter(x => x > 2);         // [3, 4, 5]
arr.reduce((sum, x) => sum + x, 0); // 15
arr.find(x => x > 2);           // 3
arr.findIndex(x => x > 2);      // 2
arr.some(x => x > 3);           // true
arr.every(x => x > 0);          // true

// Other methods
arr.includes(3);                // true
arr.indexOf(3);                 // 2
arr.join(", ");                 // "1, 2, 3, 4, 5"
arr.reverse();                  // Reverse array
arr.sort();                     // Sort array
arr.concat([6, 7]);             // [1, 2, 3, 4, 5, 6, 7]
[...arr];                       // Spread operator (copy array)
```

### Objects
```javascript
let person = {
    name: "John",
    age: 30,
    greet: function() {
        return `Hello, I'm ${this.name}`;
    }
};

// Access properties
person.name;                    // "John"
person["age"];                  // 30

// Add/modify properties
person.city = "New York";
person.age = 31;

// Delete property
delete person.age;

// Object methods
Object.keys(person);            // ["name", "age", "greet"]
Object.values(person);          // ["John", 30, function]
Object.entries(person);         // [["name", "John"], ["age", 30]]
Object.assign({}, person);      // Copy object
{ ...person };                  // Spread operator (copy)

// Destructuring
const { name, age } = person;
```

### Functions
```javascript
// Function declaration
function greet(name) {
    return `Hello, ${name}`;
}

// Function expression
const greet = function(name) {
    return `Hello, ${name}`;
};

// Arrow function
const greet = (name) => `Hello, ${name}`;
const add = (a, b) => a + b;
const square = x => x * x;      // Single parameter, no parentheses

// Default parameters
function greet(name = "Guest") {
    return `Hello, ${name}`;
}

// Rest parameters
function sum(...numbers) {
    return numbers.reduce((a, b) => a + b, 0);
}
```

### Control Flow
```javascript
// If statement
if (condition) {
    // code
} else if (otherCondition) {
    // code
} else {
    // code
}

// Switch statement
switch (value) {
    case 1:
        // code
        break;
    case 2:
        // code
        break;
    default:
        // code
}

// For loop
for (let i = 0; i < 10; i++) {
    // code
}

// For...of loop (arrays)
for (let item of array) {
    // code
}

// For...in loop (objects)
for (let key in object) {
    // code
}

// While loop
while (condition) {
    // code
}

// Do...while loop
do {
    // code
} while (condition);
```

### DOM Manipulation
```javascript
// Select elements
document.getElementById("id");
document.querySelector(".class");
document.querySelectorAll("div");
document.getElementsByClassName("class");
document.getElementsByTagName("p");

// Modify content
element.textContent = "New text";
element.innerHTML = "<strong>Bold text</strong>";

// Modify attributes
element.getAttribute("href");
element.setAttribute("href", "https://example.com");
element.removeAttribute("class");
element.id = "new-id";
element.className = "new-class";

// Modify styles
element.style.color = "red";
element.style.backgroundColor = "blue";
element.style.display = "none";

// Classes
element.classList.add("active");
element.classList.remove("inactive");
element.classList.toggle("hidden");
element.classList.contains("active");

// Create and append elements
let newDiv = document.createElement("div");
newDiv.textContent = "Hello";
parent.appendChild(newDiv);
parent.insertBefore(newDiv, referenceNode);
parent.removeChild(child);
element.remove();

// Clone element
let clone = element.cloneNode(true); // true for deep clone
```

### Events
```javascript
// Add event listener
element.addEventListener("click", function(e) {
    console.log("Clicked!");
    e.preventDefault();          // Prevent default action
    e.stopPropagation();         // Stop event bubbling
});

// Common events
click, dblclick, mouseenter, mouseleave, mousemove
keydown, keyup, keypress
submit, change, input, focus, blur
load, resize, scroll

// Remove event listener
element.removeEventListener("click", handlerFunction);

// Event delegation
parent.addEventListener("click", function(e) {
    if (e.target.matches(".child")) {
        // Handle click on child
    }
});
```

### Async JavaScript

#### Callbacks
```javascript
function fetchData(callback) {
    setTimeout(() => {
        callback("Data loaded");
    }, 1000);
}

fetchData((data) => {
    console.log(data);
});
```

#### Promises
```javascript
let promise = new Promise((resolve, reject) => {
    setTimeout(() => {
        resolve("Success!");
        // or reject("Error!");
    }, 1000);
});

promise
    .then(result => console.log(result))
    .catch(error => console.error(error))
    .finally(() => console.log("Done"));

// Promise chaining
fetch("https://api.example.com/data")
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));
```

#### Async/Await
```javascript
async function fetchData() {
    try {
        let response = await fetch("https://api.example.com/data");
        let data = await response.json();
        return data;
    } catch (error) {
        console.error(error);
    }
}

// Using async function
fetchData().then(data => console.log(data));
```

### Fetch API
```javascript
// GET request
fetch("https://api.example.com/data")
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error(error));

// POST request
fetch("https://api.example.com/data", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({ name: "John", age: 30 })
})
    .then(response => response.json())
    .then(data => console.log(data));

// With async/await
async function getData() {
    const response = await fetch("https://api.example.com/data");
    const data = await response.json();
    return data;
}
```

### Local Storage
```javascript
// Save data
localStorage.setItem("key", "value");
localStorage.setItem("user", JSON.stringify({ name: "John" }));

// Get data
let value = localStorage.getItem("key");
let user = JSON.parse(localStorage.getItem("user"));

// Remove data
localStorage.removeItem("key");

// Clear all
localStorage.clear();

// Session storage (same API, cleared when tab closes)
sessionStorage.setItem("key", "value");
```

### ES6+ Features
```javascript
// Destructuring
const [a, b] = [1, 2];
const { name, age } = person;

// Spread operator
const arr2 = [...arr1, 4, 5];
const obj2 = { ...obj1, city: "NYC" };

// Rest parameters
function sum(...numbers) { }

// Template literals
const str = `Hello ${name}`;

// Arrow functions
const add = (a, b) => a + b;

// Classes
class Person {
    constructor(name) {
        this.name = name;
    }
    greet() {
        return `Hello, ${this.name}`;
    }
}

// Modules
export const name = "John";
import { name } from "./module.js";

// Optional chaining
const street = user?.address?.street;

// Nullish coalescing
const name = username ?? "Guest";
```

---

## Popular Libraries & Frameworks

### React (Library)
```javascript
// Component
function Welcome({ name }) {
    return <h1>Hello, {name}</h1>;
}

// Hooks
import { useState, useEffect } from 'react';

function Counter() {
    const [count, setCount] = useState(0);
    
    useEffect(() => {
        document.title = `Count: ${count}`;
    }, [count]);
    
    return (
        <button onClick={() => setCount(count + 1)}>
            Count: {count}
        </button>
    );
}
```

### Vue (Framework)
```vue
<template>
    <div>
        <h1>{{ message }}</h1>
        <button @click="increment">Count: {{ count }}</button>
    </div>
</template>

<script>
export default {
    data() {
        return {
            message: 'Hello Vue',
            count: 0
        }
    },
    methods: {
        increment() {
            this.count++;
        }
    }
}
</script>
```

### Tailwind CSS (Utility-first CSS)
```html
<div class="flex items-center justify-center h-screen bg-gray-100">
    <button class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
        Click me
    </button>
</div>
```

### Bootstrap (CSS Framework)
```html
<div class="container">
    <div class="row">
        <div class="col-md-6">
            <button class="btn btn-primary">Primary Button</button>
        </div>
    </div>
</div>
```

---

## Tools & Commands

### NPM (Node Package Manager)
```bash
npm init                        # Initialize project
npm install package-name        # Install package
npm install -g package-name     # Install globally
npm install --save-dev package  # Install as dev dependency
npm update                      # Update packages
npm uninstall package-name      # Remove package
npm run script-name             # Run script from package.json
```

### Git
```bash
git init                        # Initialize repository
git clone <url>                 # Clone repository
git add .                       # Stage all changes
git commit -m "message"         # Commit changes
git push origin main            # Push to remote
git pull                        # Pull from remote
git branch branch-name          # Create branch
git checkout branch-name        # Switch branch
git merge branch-name           # Merge branch
git status                      # Check status
git log                         # View commit history
```

### Browser DevTools
```
F12 or Ctrl+Shift+I             # Open DevTools
Console tab                     # JavaScript console
Elements tab                    # Inspect HTML/CSS
Network tab                     # Monitor requests
Sources tab                     # Debug JavaScript
Application tab                 # Storage, cache, etc.
```

---

## Best Practices

### Performance
- Minimize HTTP requests
- Optimize images (compress, use appropriate formats)
- Minify CSS, JavaScript files
- Use lazy loading for images and components
- Implement caching strategies
- Use CDN for static assets
- Defer or async load non-critical scripts

### Accessibility
- Use semantic HTML elements
- Add alt text to images
- Ensure keyboard navigation works
- Use proper heading hierarchy (h1-h6)
- Maintain sufficient color contrast
- Add ARIA labels when needed
- Test with screen readers

### SEO
- Use descriptive title tags
- Write meaningful meta descriptions
- Use semantic HTML5 elements
- Optimize images with alt text
- Create XML sitemap
- Ensure mobile responsiveness
- Improve page load speed

### Code Organization
- Use consistent naming conventions
- Comment complex code
- Keep functions small and focused
- Follow DRY principle (Don't Repeat Yourself)
- Use version control (Git)
- Write modular, reusable code
- Separate concerns (HTML, CSS, JS)
