import Link from "next/link";

const Navbar = () => (
  <>
    <h3>Links</h3>
    <ul>
      <li><Link href='/'>Face search page</Link></li>
      <li><Link href='/opencv'>opencv test page</Link></li>
    </ul>
  </>
)

export default Navbar;
