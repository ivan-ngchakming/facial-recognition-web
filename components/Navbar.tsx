import Link from "next/link";

const Navbar = () => (
  <>
    <h3>Links</h3>
    <ul>
      <li><Link href='/'>Face search page</Link></li>
      <li><Link href='/opencv'>Face detect (Image)</Link></li>
      <li><Link href='/opencv/video'>Face detect (Video)</Link></li>
    </ul>
  </>
)

export default Navbar;
