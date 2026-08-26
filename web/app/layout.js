import "./globals.css";

export const metadata = {
  title: "Video Dialogue Finder",
  description: "Find the exact frame where a line of dialogue first appears in a video.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
