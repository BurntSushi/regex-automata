/*!
TODO
*/

// BREADCRUMBS: Build the higher level regex API here. Move the iterators
// from vm.rs to here. We'll need another layer of Config/Builder. Then strip
// down vm.rs to just its basic find routines without any iterators. Other
// than some vague high vs low level difference, the regex API will permit
// bundling a prefilter where as the VM layer will require explicitly providing
// a prefilter scanner to each search call.
