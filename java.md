# Java Cheatsheet

## Basic Syntax

### Hello World
```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

### Comments
```java
// Single-line comment

/* Multi-line
   comment */

/** JavaDoc comment
  * @param name description
  * @return description
  */
```

## Data Types

### Primitive Types
```java
byte b = 127;              // 8-bit: -128 to 127
short s = 32767;           // 16-bit: -32,768 to 32,767
int i = 2147483647;        // 32-bit: -2^31 to 2^31-1
long l = 9223372036854775807L; // 64-bit: -2^63 to 2^63-1

float f = 3.14f;           // 32-bit floating point
double d = 3.14159;        // 64-bit floating point

char c = 'A';              // 16-bit Unicode character
boolean bool = true;       // true or false
```

### Reference Types
```java
String str = "Hello";
int[] array = {1, 2, 3};
ArrayList<Integer> list = new ArrayList<>();
```

## Variables

### Declaration and Initialization
```java
int x;                     // Declaration
x = 10;                    // Assignment
int y = 20;                // Declaration + initialization
final int CONSTANT = 100;  // Constant (cannot be changed)
```

### Type Conversion
```java
// Implicit (widening)
int i = 100;
double d = i;

// Explicit (narrowing)
double d = 9.78;
int i = (int) d;  // i = 9
```

## Operators

### Arithmetic
```java
int a = 10, b = 3;
int sum = a + b;       // 13
int diff = a - b;      // 7
int prod = a * b;      // 30
int quot = a / b;      // 3
int rem = a % b;       // 1
```

### Comparison
```java
== // Equal to
!= // Not equal to
>  // Greater than
<  // Less than
>= // Greater than or equal to
<= // Less than or equal to
```

### Logical
```java
&& // AND
|| // OR
!  // NOT
```

### Assignment
```java
=   // Assign
+=  // Add and assign
-=  // Subtract and assign
*=  // Multiply and assign
/=  // Divide and assign
%=  // Modulus and assign
++  // Increment
--  // Decrement
```

## Control Flow

### If-Else
```java
if (condition) {
    // code
} else if (anotherCondition) {
    // code
} else {
    // code
}
```

### Ternary Operator
```java
int result = (condition) ? valueIfTrue : valueIfFalse;
```

### Switch
```java
switch (variable) {
    case value1:
        // code
        break;
    case value2:
        // code
        break;
    default:
        // code
}
```

### For Loop
```java
for (int i = 0; i < 10; i++) {
    // code
}

// Enhanced for loop (for-each)
for (Type element : collection) {
    // code
}
```

### While Loop
```java
while (condition) {
    // code
}
```

### Do-While Loop
```java
do {
    // code
} while (condition);
```

### Loop Control
```java
break;      // Exit loop
continue;   // Skip to next iteration
return;     // Exit method
```

## Arrays

### Declaration and Initialization
```java
int[] arr1 = new int[5];
int[] arr2 = {1, 2, 3, 4, 5};
int[][] matrix = new int[3][3];
int[][] matrix2 = {{1, 2}, {3, 4}};
```

### Array Operations
```java
int length = arr.length;
arr[0] = 10;              // Access element
Arrays.sort(arr);         // Sort array
Arrays.toString(arr);     // Convert to string
Arrays.copyOf(arr, length);
```

## Strings

### String Methods
```java
String str = "Hello World";

str.length();              // 11
str.charAt(0);             // 'H'
str.substring(0, 5);       // "Hello"
str.toLowerCase();         // "hello world"
str.toUpperCase();         // "HELLO WORLD"
str.trim();                // Remove whitespace
str.replace("World", "Java"); // "Hello Java"
str.split(" ");            // ["Hello", "World"]
str.equals("Hello");       // false
str.equalsIgnoreCase("hello world"); // true
str.contains("World");     // true
str.startsWith("Hello");   // true
str.indexOf("o");          // 4
```

### String Concatenation
```java
String s1 = "Hello";
String s2 = "World";
String s3 = s1 + " " + s2;  // "Hello World"
String s4 = String.format("%s %s", s1, s2);
```

### StringBuilder
```java
StringBuilder sb = new StringBuilder();
sb.append("Hello");
sb.append(" World");
sb.insert(5, ",");
sb.delete(5, 6);
sb.reverse();
String result = sb.toString();
```

## Classes and Objects

### Class Definition
```java
public class Person {
    // Fields (instance variables)
    private String name;
    private int age;
    
    // Constructor
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    // Getter
    public String getName() {
        return name;
    }
    
    // Setter
    public void setName(String name) {
        this.name = name;
    }
    
    // Method
    public void greet() {
        System.out.println("Hello, I'm " + name);
    }
    
    // Static method
    public static void staticMethod() {
        // code
    }
}
```

### Object Creation
```java
Person person = new Person("Alice", 25);
person.greet();
String name = person.getName();
```

## Inheritance

```java
public class Animal {
    protected String name;
    
    public void eat() {
        System.out.println("Eating...");
    }
}

public class Dog extends Animal {
    public void bark() {
        System.out.println("Woof!");
    }
    
    @Override
    public void eat() {
        System.out.println("Dog is eating...");
    }
}
```

## Interfaces

```java
public interface Drawable {
    void draw();  // Abstract method
    
    default void display() {  // Default method
        System.out.println("Displaying...");
    }
    
    static void info() {  // Static method
        System.out.println("Drawable interface");
    }
}

public class Circle implements Drawable {
    @Override
    public void draw() {
        System.out.println("Drawing circle");
    }
}
```

## Abstract Classes

```java
public abstract class Shape {
    abstract void draw();  // Abstract method
    
    public void display() {  // Concrete method
        System.out.println("Displaying shape");
    }
}

public class Rectangle extends Shape {
    @Override
    void draw() {
        System.out.println("Drawing rectangle");
    }
}
```

## Collections

### ArrayList
```java
ArrayList<String> list = new ArrayList<>();
list.add("Apple");
list.add("Banana");
list.get(0);              // "Apple"
list.set(0, "Orange");
list.remove(0);
list.size();
list.contains("Apple");
list.clear();
```

### HashMap
```java
HashMap<String, Integer> map = new HashMap<>();
map.put("Alice", 25);
map.put("Bob", 30);
map.get("Alice");         // 25
map.remove("Bob");
map.containsKey("Alice"); // true
map.containsValue(25);    // true
map.size();
map.keySet();             // Set of keys
map.values();             // Collection of values
```

### HashSet
```java
HashSet<String> set = new HashSet<>();
set.add("Apple");
set.add("Banana");
set.add("Apple");         // Won't be added (duplicates not allowed)
set.contains("Apple");    // true
set.remove("Banana");
set.size();
```

### LinkedList
```java
LinkedList<String> list = new LinkedList<>();
list.add("First");
list.addFirst("Start");
list.addLast("End");
list.removeFirst();
list.removeLast();
```

## Exception Handling

```java
try {
    // Code that might throw exception
    int result = 10 / 0;
} catch (ArithmeticException e) {
    // Handle specific exception
    System.out.println("Cannot divide by zero");
} catch (Exception e) {
    // Handle any other exception
    System.out.println("Error: " + e.getMessage());
} finally {
    // Always executes
    System.out.println("Cleanup code");
}
```

### Throwing Exceptions
```java
public void checkAge(int age) throws IllegalArgumentException {
    if (age < 0) {
        throw new IllegalArgumentException("Age cannot be negative");
    }
}
```

## File I/O

### Reading a File
```java
try {
    File file = new File("input.txt");
    Scanner scanner = new Scanner(file);
    while (scanner.hasNextLine()) {
        String line = scanner.nextLine();
        System.out.println(line);
    }
    scanner.close();
} catch (FileNotFoundException e) {
    System.out.println("File not found");
}
```

### Writing to a File
```java
try {
    FileWriter writer = new FileWriter("output.txt");
    writer.write("Hello, World!\n");
    writer.close();
} catch (IOException e) {
    System.out.println("Error writing file");
}
```

## Lambda Expressions

```java
// Syntax: (parameters) -> expression
// or: (parameters) -> { statements; }

// Example with List
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
numbers.forEach(n -> System.out.println(n));

// Example with filter
numbers.stream()
       .filter(n -> n % 2 == 0)
       .forEach(System.out::println);
```

## Streams

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);

// Filter
List<Integer> evens = numbers.stream()
                             .filter(n -> n % 2 == 0)
                             .collect(Collectors.toList());

// Map
List<Integer> squared = numbers.stream()
                               .map(n -> n * n)
                               .collect(Collectors.toList());

// Reduce
int sum = numbers.stream()
                 .reduce(0, (a, b) -> a + b);

// Sorted
List<Integer> sorted = numbers.stream()
                              .sorted()
                              .collect(Collectors.toList());
```

## Generics

```java
// Generic class
public class Box<T> {
    private T value;
    
    public void set(T value) {
        this.value = value;
    }
    
    public T get() {
        return value;
    }
}

Box<Integer> intBox = new Box<>();
intBox.set(10);

// Generic method
public <T> void printArray(T[] array) {
    for (T element : array) {
        System.out.println(element);
    }
}
```

## Enums

```java
public enum Day {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
}

Day today = Day.MONDAY;

switch (today) {
    case MONDAY:
        System.out.println("Start of work week");
        break;
    case FRIDAY:
        System.out.println("Almost weekend");
        break;
    default:
        System.out.println("Another day");
}
```

## Common Annotations

```java
@Override          // Method overrides superclass method
@Deprecated        // Method/class is deprecated
@SuppressWarnings  // Suppress compiler warnings
@FunctionalInterface // Interface has exactly one abstract method
```

## Useful Methods

### Math Class
```java
Math.max(a, b);
Math.min(a, b);
Math.abs(x);
Math.pow(base, exponent);
Math.sqrt(x);
Math.random();  // Random number between 0.0 and 1.0
Math.round(x);
Math.floor(x);
Math.ceil(x);
```

### Object Methods
```java
obj.toString();    // String representation
obj.equals(other); // Check equality
obj.hashCode();    // Hash code value
obj.getClass();    // Class object
```
